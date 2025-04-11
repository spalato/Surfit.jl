using Surfit: hash_call

function print_test_hashes()
    test_cases = [
        ("f", (1, 2, 3)),
        ("f", (2, 2, 3)), # slightly different values
        ("f", (4.5, 6.7)),
        ("f", (4.5+eps(4.5), 6.7)), # slightly different values
        ("f", (4.5, 6.7, 0.0))
    ]
    
    for (name, p) in test_cases
        hash = hash_call(p)
        println("(\"$name\", $p, \"$hash\"),")
    end
end
