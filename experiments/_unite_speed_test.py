import nama
from cProfile import run

matcher = nama.Matcher([f'string_{i:010d}' for i in range(100000)])

m1 = matcher.unite([f'string_{i:010d}' for i in range(50000)])

run('m1.unite(lambda s:s[-3:])',sort='tottime')


m2 = m1.unite(lambda s:s[-3:])
#
# # m1 = matcher.unite(lambda s:s[:4])
# m2 = m1.unite(lambda s:s[-3:])
#
# min([(-1,'a'),(-4,'b'),(-1,'b'),(-4,'a')])
#
# matcher.to_df()
