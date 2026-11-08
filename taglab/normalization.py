"""Exact finite fixed-score global-vs-local normalization witness, not training."""
from fractions import Fraction


def comparison(weight=Fraction(1,9)):
    if not isinstance(weight,Fraction) or weight<=0:raise ValueError('positive Fraction required')
    # Three paths: A/C, B/A, B/C. A has one successor and B has two.
    paths=[['A','C'],['B','A'],['B','C']];raw=[weight,Fraction(1),weight];z=sum(raw)
    global_p=[v/z for v in raw]
    local_p=[Fraction(1,2),Fraction(1,2)/(1+weight),Fraction(1,2)*weight/(1+weight)]
    return dict(paths=paths,raw_weights=[str(x) for x in raw],global_partition=str(z),globally_normalized=[str(x) for x in global_p],locally_normalized=[str(x) for x in local_p],global_branch_A=str(global_p[0]),local_branch_A=str(local_p[0]))
