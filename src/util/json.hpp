// A small JSON reader (RFC 8259) for settings files: parses a whole text into
// a tree of values.  Strict - no comments, no trailing commas, no repeated
// keys - but it says on which line a mistake is.
#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace spectro {

struct JsonMember;

struct JsonValue {
    enum class Type { Null, Boolean, Number, String, Array, Object };

    Type                    type    = Type::Null;
    bool                    boolean = false;
    double                  number  = 0.0;
    std::string             string;
    std::vector<JsonValue>  items;    // an array's
    std::vector<JsonMember> members;  // an object's, in the order written
    int                     line = 1; // where the value starts in the text

    // An object's member called `key`; nullptr if there is none.
    const JsonValue* find(std::string_view key) const;
};

struct JsonMember {
    std::string first;
    JsonValue   second;
};

// "a number", "an object", ...: for messages.
const char* json_type_name(JsonValue::Type type);

// Parses a JSON text.  On failure returns false, and says what is wrong and
// where in `error` ("line 3: expected ',' or ']'").
bool parse_json(std::string_view text, JsonValue& value, std::string& error);

}  // namespace spectro
