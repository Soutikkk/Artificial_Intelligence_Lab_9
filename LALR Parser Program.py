
# LALR(1) Parser Implementation

from collections import defaultdict

# Grammar
# E -> E + T | T
# T -> T * F | F
# F -> ( E ) | id

productions = [
    ("S'", ["E"]),
    ("E", ["E", "+", "T"]),
    ("E", ["T"]),
    ("T", ["T", "*", "F"]),
    ("T", ["F"]),
    ("F", ["(", "E", ")"]),
    ("F", ["id"])
]

# Parsing table (ACTION and GOTO)
ACTION = {
    0: {"id": "s5", "(": "s4"},
    1: {"+": "s6", "$": "acc"},
    2: {"+": "r2", "*": "s7", ")": "r2", "$": "r2"},
    3: {"+": "r4", "*": "r4", ")": "r4", "$": "r4"},
    4: {"id": "s5", "(": "s4"},
    5: {"+": "r6", "*": "r6", ")": "r6", "$": "r6"},
    6: {"id": "s5", "(": "s4"},
    7: {"id": "s5", "(": "s4"},
    8: {"+": "s6", ")": "s11"},
    9: {"+": "r1", "*": "s7", ")": "r1", "$": "r1"},
    10: {"+": "r3", "*": "r3", ")": "r3", "$": "r3"},
    11: {"+": "r5", "*": "r5", ")": "r5", "$": "r5"}
}

GOTO = {
    0: {"E": 1, "T": 2, "F": 3},
    4: {"E": 8, "T": 2, "F": 3},
    6: {"T": 9, "F": 3},
    7: {"F": 10}
}

def lalr_parser(input_string):
    tokens = input_string.split() + ["$"]
    stack = [0]
    index = 0

    print("\nSTACK\t\tINPUT\t\tACTION")
    print("-" * 50)

    while True:
        state = stack[-1]
        symbol = tokens[index]

        action = ACTION.get(state, {}).get(symbol)

        print(stack, "\t", tokens[index:], "\t", action)

        if action is None:
            print("\nString Rejected!")
            return

        if action == "acc":
            print("\nString Accepted!")
            return

        if action.startswith("s"):
            stack.append(symbol)
            stack.append(int(action[1:]))
            index += 1

        elif action.startswith("r"):
            production_no = int(action[1:])
            lhs, rhs = productions[production_no]

            for _ in range(2 * len(rhs)):
                stack.pop()

            state = stack[-1]
            stack.append(lhs)
            stack.append(GOTO[state][lhs])

# Main program
string = input("Enter input string (use spaces): ")
lalr_parser(string)
