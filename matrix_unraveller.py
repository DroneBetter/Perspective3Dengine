dbg=(lambda x,*s: (x,print(*s,x))[0]) #debug
from functools import reduce #I will never import anything else from functools
construce=(lambda f,l,i=None: reduce(lambda a,b: f(*a,b),l,i))
from itertools import starmap,accumulate,groupby,product,combinations,chain,islice
redumulate=(lambda f,l,i=None: accumulate(l,f,initial=i))
tarmap=(lambda f,*l: tuple(starmap(f,*l)))
larmap=(lambda f,*l: list(starmap(f,*l)))
lap=(lambda func,*iterables: list(map(func,*iterables)))
tap=(lambda func,*iterables: tuple(map(func,*iterables)))
bind=(lambda *f: lambda *a: reduce(lambda a,f: (lambda a,i,f: (f(a) if i else f(*a)))(a,*f),enumerate(f),a))
transpose=(lambda l: zip(*l))
from operator import __add__,__neg__,__mul__ #for purposes of unknown-type operations
from math import isqrt
moddiv=(lambda a,b: divmod(a,b)[::-1])
from copy import deepcopy

letters=(lambda l: tap(chr,range(97,97+l)))
symbols=('-','*','+','()','lambda',',') #do not move (- is negation, and it is more compilation order than precedence)
#average=(lambda a,b: '#'+''.join(map(lambda i: (lambda a,b: (lambda n: ''.join(map(lambda i: (lambda n: chr(n+48+39*(n>9)))(n>>(2+~i<<2)&(1<<4)-1),range(2))))(isqrt((a**2+b**2)//2)))(*map(lambda s: sum(__import__('itertools').starmap(lambda i,c: (lambda d: d-48-7*(d>64)-32*(d>96))(ord(c))<<(2+~i<<2),enumerate(s[2*i+1:2*i+3]))),(a,b))),range(3))))

strucget=(lambda struc,inds: reduce(lambda a,b: a[b],inds,struc))
def strucset(struc,inds,val): #very suspicious
    if len(inds):
        strucget(struc,inds[:-1])[inds[-1]]=val
    elif type(val)==str:
        struc=val
    else:
        struc[:]=val
    return(struc)

'''(a,b, (e,f, (a*e+b*g,a*f+b*h
 c,d)* g,h)= c*e+d*g,c*f+d*h)

(a[0],a[1], (b[0],b[1], (a[0]*b[0]+a[1]*b[2],a[0]*b[1]+a[1]*b[3]
 a[2],a[3])* b[2],b[3])= a[2]*b[0]+a[3]*b[2],a[2]*b[1]+a[3]*b[3])'''
#those beginning at 0 are named sub-O(n**3) ones, algorithm -n is schoolbook for n*n (so -1 represents numerical multiplication)
names=('strassen','strinograd') #Winograd made quite a few, however including both the original D2_x 18-addition Strassen's algorithm and his 15-addition reduction for posterity purposes #thank you Winograd
size=(lambda a: -a if a<0 else (2,2)[a])
A007814=(lambda n: (~n&n-1).bit_length()) #thank you Chai Wah Wu
def unraveller(n,algs=None):
    if algs==None:
        strassens=A007814(n)
        algs=(lambda n: (-n,)*(n!=1))(n>>strassens)+(0,)*strassens
    print('n',n,algs)
    def structrans(tree,arm,leaf=None,node=None): #similar to the ones in DroneLambda (but nicer :-)
        while True:
            while type(strucget(tree,arm))==list:
                arm.append(0)
            if leaf!=None: #iterates over leaves
                tree=leaf(tree,arm)
                if not arm:
                    break
            while True:
                arm[-1]+=1
                if arm[-1]<len(strucget(tree,arm[:-1])):
                    break
                else:
                    if node!=None: #iterates over nodes
                        tree=node(tree,arm)
                    del(arm[-1])
                    if not arm:
                        break #my feeling when no do-while
            if not arm:
                break
    def enmax(tree,leaf=None,node=None,lim=0):
        global iterations,same,lambdad,lambdads
        arm=[]
        iterations=0
        lambdads=0 #rams who are parents
        same=False
        while iterations<-lim or not same if lim<0 else iterations<lim if lim else not same:
            lambdad=False
            trold=deepcopy(tree)
            structrans(tree,arm,leaf,node)
            iterations+=1
            lambdads+=lambdad
            same=tree==trold
        return(tree)
    #structure formatted with function applied at end ('*' and '+' are multiply and add (including matrix thereof), '()' is call, '[]' is index ('][' of which has non-square uppermost  (for things like Strassen's algorithm that don't want to be disturbed)), only 'lambda' is preceding, ',' is ravel (slightly similarly to APL))
    tree=['*',['[]',0],['[]',1]] #indexes (first element is operand, all thereafter are flatly-encoded coordinates (in their own layer of nestedness) to be recombined)
    leaves=[[]]
    nest=(lambda n,i,d=False: (n if n[0] in {'[]',']['} else ['[]',n])+(i if d else [i]))
    unfold=(lambda blocktrix,inner,outer,ii=False: [',']+(blocktrix if outer==1 else lap(lambda i: (lambda y,x: ['[]'[::(-1)**ii],blocktrix[x[1]+y[1]*inner],(x[0]+y[0]*outer)])(*map(lambda j: moddiv(j,inner),i)),product(range(outer**2),repeat=2))))
    block=n
    for a in algs:
        block//=size(a)
        for l in leaves:
            if a<0:
                strucset(tree,l,(lambda t: (lambda u: u if block==1 else ['()',['lambda',tap(lambda i: 'm'+str(i),range(a**2)),':',unfold(lap(lambda i: 'm'+str(i),range(a**2)),-a,block)],u])([',']+larmap(lambda y,x: ['+']+lap(lambda i: ['*',nest(t[1],i-a*y),nest(t[2],x-a*i)],range(-a)),product(range(-a),repeat=2))))(strucget(tree,l)))
            else:
                if a==0:
                    strucset(tree,l,(lambda a,b: ['()',['lambda',tap(lambda i: 'm'+str(i),range(7)),':',unfold([['+','m0','m3',['-','m4'],'m6'],['+','m2','m4'],['+','m1','m3'],['+','m0',['-','m1'],'m2','m5']],2,block)],[',',['*',['+',nest(a,0),nest(a,3)],['+',nest(b,0),nest(b,3)]],['*',['+',nest(a,2),nest(a,3)],nest(b,0)],['*',nest(a,0),['+',nest(b,1),['-',nest(b,3)]]],['*',nest(a,3),['+',nest(b,2),['-',nest(b,0)]]],['*',['+',nest(a,0),nest(a,1)],nest(b,3)],['*',['+',nest(a,2),['-',nest(a,0)]],['+',nest(b,0),nest(b,1)]],['*',['+',nest(a,1),['-',nest(a,3)]],['+',nest(b,2),nest(b,3)]]]])(*strucget(tree,l)[1:]))
        if a<0:
            leaves=larmap(lambda l,i,j: l+[2]*(block!=1)+[i+1,j+1],product(leaves,range(a**2),range(-a))) #squaring removes signs (thank you 2 dimensions)
        else:
            if a==0:
                leaves=larmap(lambda l,i: l+[2,i+1],product(leaves,range(7)))
    def unraveldex(tree,arm):
        if strucget(tree,arm)=='[]' and (lambda n: type(n)==list and n[0] in {'+','-'})(strucget(tree,arm[:-1])[arm[-1]+1]):
            del(arm[-1])
            strucset(tree,arm,(lambda t: [t[1][0]]+lap(lambda i: nest(i,t[-1]),t[1][1:]))(strucget(tree,arm)))
        return(tree)
    tree=enmax(tree,leaf=unraveldex)
    def flatten(tree,arm):
        if strucget(tree,arm)=='[]':
            del(arm[-1])
            strucset(tree,arm,(lambda i: (lambda r: ['[]',i[1],sum(map(lambda c,a,t: (lambda m,d: c*(m+r*d))(*moddiv(t,a)),redumulate(int.__mul__,map(size,algs[::-1]),1),map(size,algs[::-1]),i[:1:-1]))])(reduce(int.__mul__,map(size,algs[:1-len(i):-1]),1)))(strucget(tree,arm)))
        return(tree)
    tree=enmax(tree,leaf=flatten,lim=1)
    def deeplace(i,tree): #perhaps more accurately named deepcrement but this is more whimsical
        def place(tree,arm):
            t=strucget(tree,arm)
            if type(t)==str and t[0]=='m':
                strucset(tree,arm,t[0]+str(int(t[1:])+i))
            return(tree)
        structrans(tree,[],leaf=place)
        return(tree)
    def sumction(n):
        params=lap(lambda i: len(i[1][1]),n)
        f=['()',['lambda',tap(lambda i: 'm'+str(i),range(sum(params))),':',[',']+lap(lambda i: ['+']+lap(lambda i,n: deeplace(i,n),accumulate([0]+params),i),transpose(map(lambda i: i[1][3][1:],n)))],[',']+list(chain(*map(lambda i: i[2][1:],n)))] #the only time I will ever use accumulate
        return(f)
    def stringer(tree,arm):
        global iterations,same,lambdad,lambdads
        if iterations==0:
            if strucget(tree,arm)=='[]':
                del(arm[-1])
                strucset(tree,arm,(lambda m,i: (('b' if m else 'a') if type(m)==int else m)+'['+str(i)+']')(*strucget(tree,arm)[1:]))
        else:
            t=strucget(tree,arm)
            if t in symbols:
                ind=symbols.index(t)
                if True or ind<iterations&7:
                    n=strucget(tree,arm[:-1])[1:]
                    if t=='()':
                        if iterations&7==7 and all(map(lambda s: type(s)==str,n if algs[~lambdads]<0 and False else n[1])):
                            lambdad=True
                            del(arm[-1])
                            strucset(tree,arm,(lambda t: (lambda s: s if arm else [s])('('+','.join(t[1:])+')' if algs[~lambdads]<0 and False else '('+t[1]+')'+t[2]))(strucget(tree,arm)))
                    elif t=='-':
                        if len(n)==1:
                            n=n[0]
                            if type(n)==str:
                                del(arm[-1])
                                strucset(tree,arm,t+('('+n+')' if '+' in n else n))
                    elif t=='lambda':
                        if type(n[2])==str and all(map(lambda s: type(s)==str,n[0])):
                            del(arm[-1])
                            strucset(tree,arm,(lambda t: t[0]+' '+','.join(t[1])+t[2]+' '+t[3])(strucget(tree,arm)))
                    elif t=='+':
                        if all(map(lambda i: type(i)==list and i[0]==',',n)):
                            del(arm[-1])
                            strucset(tree,arm,[',']+lap(lambda n: ['+']+list(n),islice(transpose(n),1,None,1)))
                        if all(map(lambda i: type(i)==list and i[0]=='()',n)):
                            del(arm[-1])
                            strucset(tree,arm,sumction(n))
                        elif all(map(lambda i: type(i)==str,n)):
                            del(arm[-1])
                            strucset(tree,arm,(lambda s: s if arm else [s])(''.join(starmap(lambda i,s: '+'*(i and s[0]!='-')+('('+s+')' if any(map(lambda y: y in s,symbols[ind+1:])) else s),enumerate(n)))))
                    else:
                        if all(map(lambda i: type(i)==str,n)):
                            del(arm[-1])
                            strucset(tree,arm,(lambda s: s if arm else [s])((lambda s: '('+s+')' if t==',' else s)(t.join(map(lambda s: '('+s+')' if t=='*' and '-' in s or any(map(lambda y: y in s,symbols[ind+1:])) else s,n)))))
        return(tree)
    tree=enmax(tree,leaf=stringer,lim=-8*len(algs))
    #print('lambda a,b: '+tree[0])
    return(eval('lambda a,b: '+tree[0]))
print(unraveller(2)(tuple(range(4)),tuple(range(4,8))))
print(unraveller(3)(tuple(range(9)),tuple(range(9,18))))
print(unraveller(2,(-2,))(tuple(range(4)),tuple(range(4,8))))
print(unraveller(4)(tuple(range(16)),tuple(range(16,32))))
print(unraveller(4,(-4,))(tuple(range(16)),tuple(range(16,32))))
print(unraveller(4,(-2,-2))(tuple(range(16)),tuple(range(16,32))))
print(unraveller(4,(-2,0))(tuple(range(16)),tuple(range(16,32))))
print(unraveller(4,(0,-2))(tuple(range(16)),tuple(range(16,32))))
