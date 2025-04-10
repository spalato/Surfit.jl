using Surfit: hash_call

function print_test_hashes()
    test_cases = [
        ("function1", (1, 2, 3)),
        ("function1", (2, 2, 3)), # slightly different values
        ("function1", (4.5, 6.7)), # different names
        ("function2", (4.5, 6.7)),
    ]
    
    for (name, p) in test_cases
        hash = hash_call(name, p)
        println("(\"$name\", $p, 0x$(string(hash, base=16))),")
    end
end

cases = [
    ("function1", (1, 2, 3), 0x691dcade),
    ("function1", (2, 2, 3), 0xa221fbbd),
    ("function1", (4.5, 6.7), 0xc93388e8),
    ("function2", (4.5, 6.7), 0x291eec09),
]

for (name, params, value) in cases
    # testing against hard coded values to ensure it does not depend on julia
    # version or invokation
    @test hash_call(name, params) == value
    
end
