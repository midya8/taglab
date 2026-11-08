"""Original seeded persistent-state synthetic observations, not an NLP corpus."""
import random


def generate(seed,count):
    rng=random.Random(seed);result=[]
    probs=[[.48,.28,.14,.07,.03],[.08,.22,.4,.22,.08],[.03,.07,.14,.28,.48]]
    for _ in range(count):
        n=rng.randrange(8,13);state=rng.randrange(3);x=[];y=[]
        for t in range(n):
            if t and rng.random()>.78:state=rng.choice([j for j in range(3) if j!=state])
            y.append(state);u=rng.random();total=0
            for symbol,p in zip(range(-2,3),probs[state]):
                total+=p
                if u<total:x.append(symbol);break
        result.append(dict(x=x,y=y))
    return result


def dataset():return dict(train=generate(2026110801,24),validation=generate(2026110802,12),test=generate(2026110803,16))
