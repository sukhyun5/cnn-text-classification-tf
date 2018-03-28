def test_loop(num):
    for a in num:
        print ("a : ", a)
        yield a

num = [1, 2, 3]
tl = test_loop(num)

for aa in tl:
    print ("aa : ", aa)

