#include "util/json.hpp"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace spectro {

namespace {

constexpr int kMaxDepth = 64;  // arrays and objects within each other

bool is_digit(char c) { return c >= '0' && c <= '9'; }

void append_utf8(std::string& out, uint32_t code_point)
{
    if (code_point < 0x80) {
        out += static_cast<char>(code_point);
    } else if (code_point < 0x800) {
        out += static_cast<char>(0xC0 | (code_point >> 6));
        out += static_cast<char>(0x80 | (code_point & 0x3F));
    } else if (code_point < 0x10000) {
        out += static_cast<char>(0xE0 | (code_point >> 12));
        out += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (code_point & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (code_point >> 18));
        out += static_cast<char>(0x80 | ((code_point >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (code_point & 0x3F));
    }
}

// Recursive descent over the text; the first mistake ends it.
class JsonParser {
public:
    explicit JsonParser(std::string_view text) : text_(text) {}

    bool parse(JsonValue& value, std::string& error)
    {
        // A byte order mark, as some editors write, is not part of the text.
        if (text_.substr(0, 3) == "\xEF\xBB\xBF")
            pos_ = 3;
        if (!parse_value(value, 0) || !at_end()) {
            error = "line " + std::to_string(error_line_) + ": " + error_;
            return false;
        }
        return true;
    }

private:
    bool fail(std::string what)
    {
        error_      = std::move(what);
        error_line_ = line_;
        return false;
    }

    // What is at the current position where something else was expected.
    bool unexpected(const std::string& expected)
    {
        if (pos_ >= text_.size())
            return fail("the text ends where " + expected + " should be");
        if (text_[pos_] == '/')
            return fail("comments are not allowed in JSON");
        return fail("expected " + expected);
    }

    // Skips white space; then the next character, or 0 at the end.
    char peek()
    {
        for (; pos_ < text_.size(); ++pos_) {
            const char c = text_[pos_];
            if (c == '\n')
                ++line_;
            else if (c != ' ' && c != '\t' && c != '\r')
                return c;
        }
        return '\0';
    }

    bool at_end()
    {
        if (peek() != '\0')
            return unexpected("nothing more after the value");
        return true;
    }

    bool parse_value(JsonValue& value, int depth)
    {
        const char c = peek();
        value      = {};
        value.line = line_;
        switch (c) {
        case '{':
            return parse_object(value, depth);
        case '[':
            return parse_array(value, depth);
        case '"':
            value.type = JsonValue::Type::String;
            return parse_string(value.string);
        case 't':
        case 'f':
            value.type    = JsonValue::Type::Boolean;
            value.boolean = c == 't';
            return parse_word(value.boolean ? "true" : "false");
        case 'n':
            return parse_word("null");
        default:
            if (c == '-' || is_digit(c))
                return parse_number(value);
            return unexpected("a value");
        }
    }

    bool parse_word(std::string_view word)
    {
        const std::string_view rest = text_.substr(pos_);
        const bool             whole = rest.substr(0, word.size()) == word &&
                           (rest.size() == word.size() || !std::isalnum(static_cast<unsigned char>(rest[word.size()])));
        if (!whole)
            return unexpected("a value");
        pos_ += word.size();
        return true;
    }

    bool parse_number(JsonValue& value)
    {
        // -? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?
        const size_t start  = pos_;
        const auto   digits = [&] {
            const size_t first = pos_;
            while (pos_ < text_.size() && is_digit(text_[pos_]))
                ++pos_;
            return pos_ > first;
        };
        const auto next_is = [&](char c) { return pos_ < text_.size() && text_[pos_] == c; };
        if (next_is('-'))
            ++pos_;
        if (next_is('0'))
            ++pos_;
        else if (!digits())
            return fail("invalid number");
        if (next_is('.')) {
            ++pos_;
            if (!digits())
                return fail("invalid number");
        }
        if (next_is('e') || next_is('E')) {
            ++pos_;
            if (next_is('+') || next_is('-'))
                ++pos_;
            if (!digits())
                return fail("invalid number");
        }
        // Whatever comes next must not carry on the number ("01", "1x", "1.2.3").
        if (pos_ < text_.size() && (std::isalnum(static_cast<unsigned char>(text_[pos_])) || text_[pos_] == '.'))
            return fail("invalid number");

        // Use std::strtod rather than floating-point std::from_chars, which
        // Apple's libc++ did not ship prior to Xcode 26.
        const std::string token(text_.substr(start, pos_ - start));
        char*             end    = nullptr;
        errno                    = 0;
        const double      parsed = std::strtod(token.c_str(), &end);
        if (errno == ERANGE || !std::isfinite(parsed))
            return fail("number out of range");
        if (end != token.c_str() + token.size())
            return fail("invalid number");
        value.number = parsed;
        value.type   = JsonValue::Type::Number;
        return true;
    }

    bool parse_hex4(uint32_t& code)
    {
        if (pos_ + 4 > text_.size())
            return false;
        code = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text_[pos_++];
            code <<= 4;
            if (is_digit(c))
                code |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f')
                code |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F')
                code |= static_cast<uint32_t>(c - 'A' + 10);
            else
                return false;
        }
        return true;
    }

    bool parse_string(std::string& out)
    {
        ++pos_;  // the opening quote
        for (;;) {
            if (pos_ >= text_.size())
                return fail("unterminated string");
            const char c = text_[pos_++];
            if (c == '"')
                return true;
            if (c == '\n')
                return fail("unterminated string");
            if (static_cast<unsigned char>(c) < 0x20)
                return fail("control character in a string (write it as an escape, such as \\t)");
            if (c != '\\') {
                out += c;
                continue;
            }
            if (pos_ >= text_.size())
                return fail("unterminated string");
            switch (text_[pos_++]) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u': {
                uint32_t code = 0;
                if (!parse_hex4(code))
                    return fail("invalid \\u escape in a string");
                if (code >= 0xD800 && code <= 0xDBFF) {
                    // The first half of a surrogate pair: the second must follow.
                    uint32_t low = 0;
                    if (text_.substr(pos_, 2) != "\\u")
                        return fail("invalid \\u escape in a string");
                    pos_ += 2;
                    if (!parse_hex4(low) || low < 0xDC00 || low > 0xDFFF)
                        return fail("invalid \\u escape in a string");
                    code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                } else if (code >= 0xDC00 && code <= 0xDFFF) {
                    return fail("invalid \\u escape in a string");
                }
                append_utf8(out, code);
                break;
            }
            default:
                return fail("invalid escape in a string");
            }
        }
    }

    bool parse_array(JsonValue& value, int depth)
    {
        if (depth >= kMaxDepth)
            return fail("nested too deeply");
        value.type = JsonValue::Type::Array;
        ++pos_;  // '['
        if (peek() == ']') {
            ++pos_;
            return true;
        }
        for (;;) {
            JsonValue item;
            if (!parse_value(item, depth + 1))
                return false;
            value.items.push_back(std::move(item));
            const char c = peek();
            if (c == ']') {
                ++pos_;
                return true;
            }
            if (c != ',')
                return unexpected("',' or ']'");
            ++pos_;
            if (peek() == ']')
                return fail("a comma before ']': JSON allows no trailing comma");
        }
    }

    bool parse_object(JsonValue& value, int depth)
    {
        if (depth >= kMaxDepth)
            return fail("nested too deeply");
        value.type = JsonValue::Type::Object;
        ++pos_;  // '{'
        if (peek() == '}') {
            ++pos_;
            return true;
        }
        for (;;) {
            if (peek() != '"')
                return unexpected("a key (a string in double quotes)");
            std::string key;
            if (!parse_string(key))
                return false;
            if (value.find(key))
                return fail("the key \"" + key + "\" is given twice");
            if (peek() != ':')
                return unexpected("':' after the key");
            ++pos_;
            JsonValue member;
            if (!parse_value(member, depth + 1))
                return false;
            value.members.push_back({std::move(key), std::move(member)});
            const char c = peek();
            if (c == '}') {
                ++pos_;
                return true;
            }
            if (c != ',')
                return unexpected("',' or '}'");
            ++pos_;
            if (peek() == '}')
                return fail("a comma before '}': JSON allows no trailing comma");
        }
    }

    std::string_view text_;
    size_t           pos_  = 0;
    int              line_ = 1;
    std::string      error_;
    int              error_line_ = 1;
};

}  // namespace

const JsonValue* JsonValue::find(std::string_view key) const
{
    for (const auto& [name, value] : members) {
        if (name == key)
            return &value;
    }
    return nullptr;
}

const char* json_type_name(JsonValue::Type type)
{
    switch (type) {
    case JsonValue::Type::Null: return "null";
    case JsonValue::Type::Boolean: return "true or false";
    case JsonValue::Type::Number: return "a number";
    case JsonValue::Type::String: return "a string";
    case JsonValue::Type::Array: return "an array";
    case JsonValue::Type::Object: return "an object";
    }
    return "a value";
}

bool parse_json(std::string_view text, JsonValue& value, std::string& error)
{
    return JsonParser(text).parse(value, error);
}

}  // namespace spectro
