from fuzzy.fuzzy_reasoning_v5 import FuzzyReasoning

def approx_equal(a,b,eps=1e-9):
    return abs(a-b) < eps

def test_basic_cases():
    fr = FuzzyReasoning()
    r1 = fr.calculate_fuzzy(input_nodes={30000000:0.5,30000001:0.5}, output_node=20000000)
    # expected 0.625 from manual calculation
    assert approx_equal(r1, 0.625)

    r2 = fr.calculate_fuzzy(input_nodes={30000000:0.2,30000001:0.9}, output_node=20000000)
    # expected 0.56 from manual calculation
    assert approx_equal(r2, 0.56)

    r3 = fr.calculate_fuzzy(input_nodes={30000000:0.2,30000001:0.9}, output_node=20000000, t_norm='min')
    # expected 0.583333333... from manual min-norm calculation
    assert approx_equal(r3, 0.5833333333333333)

if __name__=="__main__":
    test_basic_cases()
    print('All tests passed')
