"""A small flag parser: `--name value` pairs plus boolean `--flag`s."""

from std.collections import Dict


struct Args(Movable):
    var command: String
    var values: Dict[String, String]
    var flags: List[String]
    var positional: List[String]
    var unknown: List[String]

    def __init__(
        out self,
        argv: List[String],
        value_flags: List[String],
        boolean_flags: List[String],
    ) raises:
        """Parses `argv[2:]` against the flags the command declares.

        A flag the command does not declare goes to `unknown` and takes no
        value. `--help` and `-h` set the boolean flag `help` for every command.
        """
        self.values = Dict[String, String]()
        self.flags = List[String]()
        self.positional = List[String]()
        self.unknown = List[String]()
        self.command = argv[1] if len(argv) > 1 else String("")
        var i = 2
        while i < len(argv):
            var a = argv[i]
            if a == "--help" or a == "-h":
                self.flags.append("help")
                i += 1
            elif a.startswith("--"):
                var name = String(a[byte=2:])
                var is_bool = False
                for f in boolean_flags:
                    if f == name:
                        is_bool = True
                if is_bool:
                    self.flags.append(name)
                    i += 1
                    continue
                var takes_value = False
                for f in value_flags:
                    if f == name:
                        takes_value = True
                if not takes_value:
                    self.unknown.append(name)
                    i += 1
                    continue
                if i + 1 >= len(argv):
                    raise Error("flag --" + name + " needs a value")
                self.values[name] = argv[i + 1]
                i += 2
            else:
                self.positional.append(a)
                i += 1

    def has(self, name: String) -> Bool:
        if name in self.values:
            return True
        for f in self.flags:
            if f == name:
                return True
        return False

    def get(self, name: String) raises -> String:
        if name not in self.values:
            raise Error("missing required flag --" + name)
        return self.values[name]

    def get_or(self, name: String, default: String) raises -> String:
        if name in self.values:
            return self.values[name]
        return default

    def int_or(self, name: String, default: Int) raises -> Int:
        if name in self.values:
            return Int(self.values[name])
        return default

    def float_or(self, name: String, default: Float64) raises -> Float64:
        if name in self.values:
            return Float64(self.values[name])
        return default


def edit_distance(a: String, b: String) -> Int:
    """The Levenshtein distance between two strings, counted in bytes."""
    var x = a.as_bytes()
    var y = b.as_bytes()
    var prev = List[Int]()
    for j in range(len(y) + 1):
        prev.append(j)
    for i in range(len(x)):
        var cur = List[Int]()
        cur.append(i + 1)
        for j in range(len(y)):
            var cost = 0 if x[i] == y[j] else 1
            cur.append(min(prev[j] + cost, min(prev[j + 1], cur[j]) + 1))
        prev = cur^
    return prev[len(y)]


def suggest_flag(name: String, declared: List[String]) -> String:
    """The declared flag closest to `name` (distance at most 2), or "" if none.

    The first declared flag wins a tie.
    """
    var best = String("")
    var best_distance = 3
    for flag in declared:
        var d = edit_distance(name, flag)
        if d < best_distance:
            best_distance = d
            best = flag
    return best


def unknown_flag_error(
    command: String, name: String, declared: List[String]
) -> String:
    var text = "unknown flag --" + name + " for '" + command + "'"
    var guess = suggest_flag(name, declared)
    if guess.byte_length() > 0:
        text += " (did you mean --" + guess + "?)"
    return text + "\nsee: mojo-dllm " + command + " --help"


def parse_csv_ints(s: String) raises -> List[Int]:
    var out = List[Int]()
    for part in s.split(","):
        var p = String(part).strip()
        if p.byte_length() > 0:
            out.append(Int(p))
    return out^
