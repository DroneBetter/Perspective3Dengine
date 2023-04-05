dbg=(lambda x,*s: (x,print(*s,x))[0]) #debug
from functools import reduce #I will never import anything else from functools
construce=(lambda f,l,i=None: reduce(lambda a,b: f(*a,b),l,i))
from itertools import starmap,accumulate,groupby,product,combinations,permutations,chain,pairwise,zip_longest
rle=(lambda k: tap(lambda n: (n[0],len(tuple(n[1]))),groupby(k)))
from math import factorial,comb
fact=factorial
A007814=(lambda n: (~n&n-1).bit_length()) #thank you Chai Wah Wu
invfact=(lambda x: 0 if x==1 else (lambda t: t+ilog(x//fact(t),t+1))(A007814(x))) #requires input to be a factorial
choose=(lambda n,*k: (lambda n,*k: comb(n,*k) if len(k)==1 else fact(n)//reduce(int.__mul__,map(fact,k)) if all(map(lambda k: 0<=k,k)) else 0)(n,*k,n-sum(k)))
from itertools import combinations_with_replacement as sortduct #sortduct=(lambda n,repeat: map(lambda i: tap(n.__getitem__,i),(lambda n: redumulate(lambda k,_: shortduce(lambda k,i: ((k[i]+1,)*(i+1)+k[i+1:],k[i]==n-1),range(n),k),range(choose(n+repeat-1,n)-1),(0,)*repeat))(len(n)))) #my feeling when it already existed
redumulate=(lambda f,l,i=None: accumulate(l,f,initial=i))
tarmap=(lambda f,*l: tuple(starmap(f,*l)))
larmap=(lambda f,*l: list(starmap(f,*l)))
tap=(lambda f,*i: tuple(map(f,*i)))
lap=(lambda f,*i: list(map(f,*i)))
sap=(lambda f,*i: set(map(f,*i)))
chap=(lambda f,*i: chain.from_iterable(map(f,*i)))
compose=(lambda *f: lambda *a: reduce(lambda a,f: (lambda a,i,f: (f(a) if i else f(*a)))(a,*f),enumerate(f),a))
transpose=(lambda l: zip(*l))
from operator import __add__,__neg__,__mul__,__eq__,__or__
from math import gcd,lcm,isqrt,sqrt,cos,tan,sin,acos,asin,atan,atan2,e,tau,pi,hypot,dist
from sympy import primerange
moddiv=(lambda a,b: divmod(a,b)[::-1])
from numbers import Number
sgn=(lambda n,zerositive=False: (1 if n>0 else -1 if n<0 else zerositive) if isinstance(n,Number) else (lambda m: type(n)(tap(lambda n: n/m,n)) if 0!=m!=1 else n)(hypot(*n)))

dot=(lambda a,b: sum(map(__mul__,a,b)))
from matrix_unraveller import unraveller,strucget,strucset,structrans,enmax
A002024=(lambda n: isqrt(n<<3)+1>>1)
A002260=(lambda n,b=False: (lambda s: (lambda o: (o,s) if b else o)(n-s*(s-1)//2))(A002024(n))) #1-indexed antidiagonal coordinates
A003056=(lambda n: isqrt((n<<3)+1)-1>>1)
A002262=(lambda n,b=False: (lambda s: (lambda o: (o,s-o) if b==2 else (o,s) if b else o)(n-s*(s+1)//2))(A003056(n))) #1-indexed antidiagonal coordinates
#print('\n'.join(map(lambda f: ','.join(map(lambda n: str(f(n,True)),range(64))),(A002260,A002262))));exit()
def shortduce(f,l=None,i=None,o=None,b=None): #different ending function depending on shortcut (second tuple element is used only as whether to proceed)
    if i==None: i=next(l)
    i=(i,True)
    while True:
        if i[1]: i=(f(i[0]) if l==None else f(i[0],j))
        else: return(i[0] if b==None else b(i[0]))
    return((lambda f,i: i if f==None else f(i))(o if i[1] else b,i[0]))
rany=(lambda l: next(chain(filter(lambda i: i,l),0))) #any but reducier (however faster than reduce(__or__) (by being able to terminate on infinite sequences))

class matrix3: #flatly-encoded, implementing specific size for versor methods
    def __init__(m,*t):
        m.internal=((lambda t: (1-2*(t[2]**2+t[3]**2  ),2*(t[1]*t[2]-t[0]*t[3]),2*(t[0]*t[2]+t[1]*t[3]),
                           2*(t[1]*t[2]+t[0]*t[3]),1-2*(t[1]**2+t[3]**2  ),2*(t[2]*t[3]-t[0]*t[1]),
                           2*(t[1]*t[3]-t[0]*t[2]),2*(t[0]*t[1]+t[2]*t[3]),1-2*(t[1]**2+t[2]**2  )) if type(t)==versor else tuple(t))(*t)
                    if len(t)==1 else matrix3(t))
    __getitem__=(lambda m,i: m.internal[i])
    unravelling=unraveller(3)
    __matmul__=(lambda a,b: matrix3(matrix3.unravelling(a,b)))
    __mul__=(lambda a,b: a@b if type(b)==matrix3 else a@matrix3(b) if type(b)==versor else vector3(tap(lambda r: dot(a[3*r:3*(r+1)],b),range(3))) if type(b)==vector3 else matrix3(tap(lambda i: b*i,a)) if isinstance(b,Number) else ValueError('wibble'))
    det=(lambda m: m[0]*m[4]*m[8]-m[0]*m[5]*m[7]+m[1]*m[5]*m[6]-m[1]*m[3]*m[8]+m[2]*m[3]*m[7]-m[2]*m[4]*m[6])
    __rmul__=(lambda a,b: a*b)

class versor: #i through x (y to z), j through y (z to x), k through z (x to y) #no normalisation
    def __init__(q,*t):
        q.internal=((lambda t: (lambda u,q: ((lambda u: tap(lambda i: u*i,q))(u**-0.5/2)))(*( ( (lambda u: (u,(t[3]-t[1],-t[2]-t[6],-t[5]-t[7], u        )))(1-t[0]-t[4]+t[8]) #my feeling when I cannot fast-inverse-square-root
                                                                         if t[0]<-t[4] else
                                                                          (lambda u: (u,(u        , t[7]-t[5], t[2]-t[6], t[3]-t[1])))(1+t[0]+t[4]+t[8]))
                                                                       if t[8]>0 else
                                                                        ( (lambda u: (u,(t[7]-t[5], u        ,-t[1]-t[3],-t[2]-t[6])))(1+t[0]-t[4]-t[8])
                                                                         if t[0]>t[4] else
                                                                          (lambda u: (u,(t[6]-t[2],-t[1]-t[3], u        ,-t[5]-t[7])))(1-t[0]+t[4]-t[8])))) #from https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
                     if type(t)==matrix3 else tuple(t))(*t)
                    if len(t)==1 else versor(t))
    __getitem__=(lambda m,i: m.internal[i])
    __repr__=(lambda a: 'versor('+','.join(map(str,a.internal))+')')
    __mul__=(lambda a,b: versor((a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
                      a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
                      a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
                      a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]))
             if type(b)==versor else
              matrix3(a)*b
             if type(b) in {matrix3,vector3} else
              versor(tap(lambda i: b*i,a))
             if isinstance(b,Number) else
              ValueError('wibble'))
    __eq__=(lambda a,b: a.internal==b.internal)#(lambda a,b: all(map(__eq__,a,b)))
    def log(q):
        try:
            immag=sqrt(1-q[0]**2) #=q[1]**2+q[2]**2+q[3]**2
            coeff=acos(q[0])/immag #the q[0] in the acos would be divided by magnitude if it weren't a unit vector
        except:
            immag=0
            coeff=1 #I don't like it but it wouldn't detect float equality correctly
        return(vector3((coeff*q[1],coeff*q[2],coeff*q[3]))) #0 would be log(magnitude)
    __neg__=(lambda q: versor(map(__neg__,q)))
    __add__=(lambda a,b: versor(map(__add__,a,b)))
    __sub__=(lambda a,b: a+-b)
    conjugate=(lambda q: versor((q[0],-q[1],-q[2],-q[3])))
    __pow__=(lambda a,b: a.conjugate() if b==-1 else vector3.exp(versor.log(a)*b)) #special case can be removed if you would like more stability (living life on the edge)
    __truediv__=(lambda a,b: a*b**-1)
    canonicalise=(lambda q: sgn(q*(rany(map(sgn,filter(lambda x: abs(x)>2**-16,q)))))) #renormalise to avoid accumulating precision loss, use sgn of first nonzero (and nonerror) term

def slerp(a,b,t,x=None): #a*(b/a)**t=a*exp(log(b/a)*t), derived in https://github.com/DroneBetter/Perspective3Dengine/blob/main/perspective%203D%20engine.py
    dot=a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3]
    ang=t*acos(abs(dot))
    bc=sin(ang)*sgn(dot,True)/sqrt(1-dot**2)
    ac=cos(ang)-bc*dot
    return((ac*a[0]+bc*b[0],
            ac*a[1]+bc*b[1],
            ac*a[2]+bc*b[2],
            ac*a[3]+bc*b[3]))
def rotationParameters(a,v,w,x=None): #'in general, to rotate by amount a from some versor v to a perpendicular versor w (while conserving the perpendicular components), you need the map (lambda x: (cos(a/2)+sin(a/2)*w*v**-1)*x*(cos(a/2)+sin(a/2)*v**-1*w))'
    c=cos(a/2);s=sin(a/2)
    if x==None:
        #w*v**-1
        left =versor((c+s*( w[0]*v[0]+w[1]*v[1]+w[2]*v[2]+w[3]*v[3]),
                        s*(-w[0]*v[1]+w[1]*v[0]-w[2]*v[3]+w[3]*v[2]),
                        s*(-w[0]*v[2]+w[1]*v[3]+w[2]*v[0]-w[3]*v[1]),
                        s*(-w[0]*v[3]-w[1]*v[2]+w[2]*v[1]+w[3]*v[0])))
        #v**-1*w
        right=versor((c+s*( v[0]*w[0]+v[1]*w[1]+v[2]*w[2]+v[3]*w[3]),
                        s*( v[0]*w[1]-v[1]*w[0]-v[2]*w[3]+v[3]*w[2]),
                        s*( v[0]*w[2]+v[1]*w[3]-v[2]*w[0]-v[3]*w[1]),
                        s*( v[0]*w[3]-v[1]*w[2]+v[2]*w[1]-v[3]*w[0])))
        return(left,right)
        #left*x*right (w*v**-1*x*v**-1*w)
    else:
        b=(v[0]*w[1]-v[1]*w[0]+v[2]*w[3]-v[3]*w[2],
           v[0]*w[2]-v[1]*w[3]-v[2]*w[0]+v[3]*w[1],
           v[0]*w[3]+v[1]*w[2]-v[2]*w[1]-v[3]*w[0])
        e=(c*x[0]+s*(-b[0]*x[1]-b[1]*x[2]-b[2]*x[3]),
           c*x[1]+s*( b[0]*x[0]+b[1]*x[3]-b[2]*x[2]),
           c*x[2]+s*(-b[0]*x[3]+b[1]*x[0]+b[2]*x[1]),
           c*x[3]+s*( b[0]*x[2]-b[1]*x[1]+b[2]*x[0]))
        d=(v[0]*w[1]-v[1]*w[0]-v[2]*w[3]+v[3]*w[2],
           v[0]*w[2]+v[1]*w[3]-v[2]*w[0]-v[3]*w[1],
           v[0]*w[3]-v[1]*w[2]+v[2]*w[1]-v[3]*w[0])
        return(versor((c*e[0]+s*(-d[0]*e[1]-d[1]*e[2]-d[2]*e[3]),
                       c*e[1]+s*( d[0]*e[0]-d[1]*e[3]+d[2]*e[2]),
                       c*e[2]+s*( d[0]*e[3]+d[1]*e[0]-d[2]*e[1]),
                       c*e[3]+s*(-d[0]*e[2]+d[1]*e[1]+d[2]*e[0])))) #fastest method I have found (albeit non-composable), 64 multiplications instead of 72
class vector3:
    def __init__(v,*t):
        v.internal=tuple(t[0] if len(t)==1 else t)
    __getitem__=(lambda m,i: m.internal[i])
    __iter__=(lambda v: iter(v.internal))
    __repr__=(lambda a: 'vector3('+','.join(map(str,a.internal))+')')
    __mul__=(lambda a,b: vector3(dot(a,b)) if type(b)==vector3 else vector3(tap(lambda a: a*b,a)))
    __rmul__=(lambda a,b: a*b)
    __matmul__=(lambda a,b: vector3((a[1]*b[2]-a[2]*b[1],
                       a[2]*b[0]-a[0]*b[2],
                       a[0]*b[1]-a[1]*b[0]))) #cross
    __add__=(lambda a,b: vector3(map(__add__,a,b)))
    __neg__=(lambda v: vector3(map(__neg__,v)))
    __sub__=(lambda a,b: a+-b)
    dross=(lambda a,b: sum(a)*sum(b)-dot(a,b)) #useful in the perspective 3D engine's time mode
    abs=(lambda v: sqrt(sum(map(lambda x: x**2,v))))
    def exp(v): #meant to be specifically inverse of versor.log
        expreal=1#e**q[0]
        immag=hypot(*v) #cannot be sqrt(1-q[0]**2) due to logarithms not being unit vectors
        coeff=expreal*(immag and sin(immag)/immag)
        return(versor((expreal*cos(immag),coeff*v[0],coeff*v[1],coeff*v[2])))

def inthroot(b,n): #I hesitate to call it 'fast' integer nth root #sign-preserving (very suspicious)
    if b<0:
        return(-inthroot(-b,n))
    elif b<2 or n==1:
        return(b)
    else:
        relation=reduce(lambda r,i: tarmap(int.__sub__,pairwise((0,)+r+(0,))),range(n+1),(1,))[1:] #formerly tarmap(lambda i,n: n*(-1)**i,enumerate(reduce(lambda r,i: tarmap(int.__add__,pairwise((0,)+r+(0,))),range(n+1),(1,))[1:])) but then it came to me
        recurrence=(0,)*n
        for i in range(1,n+1):
            k=i**n
            recurrence=(k,)+recurrence
            if k>=b:
                break
        else:
            while k<b:
                i+=1
                k=dot(recurrence,relation) #pretty wacky stuff but outperforms **
                recurrence=(k,)+recurrence[:-1]
        return(i-(k>b))
mootroot=(lambda b,n: (lambda i: (b%i**n,i))(inthroot(b,n))) #like moddiv (I think it will catch on)

fratrix=(lambda m,dims=2,strict=True,hidezero=False: (lambda m: '\n'.join((lambda s: (lambda c: starmap(lambda i,r: (' ' if i else '(')+(','+'\n'*(dims==3)).join(starmap(lambda i,s: ' '*(c[i]-len(s))+s,enumerate(r)))+(',' if len(m)+~i else ')'),enumerate(s)))(tap(lambda c: max(map(len,c)),zip(*s))))(tap(lambda r: tap(lambda f: fratrix(f,2,strict) if dims==3 else str(f) if f else ' ',r),m))))(m if dims==2 else (m,)))

#print(tap(lambda n: n%inthroot(n,2)**2,range(1,256)))
#print(tarmap(lambda x,y: x%inthroot(x,y)**y,map(lambda n: A002260(n,True),range(1,256))))
#print(fratrix(tap(lambda x: tap(lambda y: x%inthroot(x,y)**y,range(1,16)),range(1,64)))) #(new sequence :-)

factorise=(lambda n: (lambda f: f+tap(lambda f: n//f,reversed(f[:-1] if isqrt(n)**2==n else f)))(tuple(filter(lambda a: n%a==0,range(1,isqrt(n)+1)))))
def stractorise(struc,inds): #structure factorise
    global diff
    if (lambda g: type(g)==int and g!=1)(strucget(struc,inds)) and (lambda g: len(g)==inds[-2]+1 or type(g[inds[-2]+1])==int)(strucget(struc,inds[:-1])):
        diff=True
        struc=strucset(struc,inds,(lambda g: [g,list(factorise(g))[1:-1]])(strucget(struc,inds)))
    return(struc,inds)
primate=(lambda n: () if n==1 else (lambda p: p if p else ((n,1),))(tuple(filter(lambda p: p[1],map(lambda p: (p,shortduce(lambda i: (i[0],False) if i[1]%p else ((i[0]+1,i[1]//p),True),i=(0,n))),reduce(lambda t,i: t+(i,)*all(map(lambda p: i%p,t)),range(2,n),()))))))
class surd:
    __repr__=(lambda a: 'surd('+''.join(starmap(lambda i,a: ('-' if sgn(a[0][0])==-1 else '+' if i else ' ')+(lambda f: f if a[1]==1 else 'sqrt('+f+')' if a[1]==2 and False else '('+f+')**(1/'+str(a[1])+')')(str(abs(a[0][0]))+('/'+str(a[0][1]))*(a[0][1]!=1)),enumerate(a.internal)))+')')
    __iter__=(lambda a: iter(a.internal))
    def simplify(a):
        while True:
            ones=[]
            cont=False
            for i,(b,e) in enumerate(a.internal):
                for j,(c,f) in enumerate(a.internal[i+1:],start=i+1):
                    if e==f and b[1]==c[1]:
                        if b[0]==-c[0]:
                            del(a.internal[j]);del(a.internal[i]);cont=True;break
                        elif b[0]==c[0]:
                            a.internal[i]=[[b[0]*2**e,b[1]],e];del(a.internal[j]);cont=True;break
                if cont:
                    break
                elif b[0]:
                    g=gcd(*b)
                    if g!=1:
                        a.internal[i][0]=[b[0]//g,b[1]//g]
                        cont=True
                        break
                    n=[b,e]
                    for f,x in primate(e):
                        for p in range(x+1):
                            if any(map(lambda b: b!=sgn(b)*abs(inthroot(b,f**p)**f**p),b)):
                                p-=1
                                break
                        if p:
                            n=[lap(lambda b: inthroot(b,f**p),n[0]),n[1]//f**p]
                            cont=True
                    a.internal[i]=n
                    if cont:
                        break
                    if e==1:
                        ones.append(i)
                else:
                    del(a.internal[i])
            if len(ones)>1:
                frac=[0,1]
                for i in ones[::-1]:
                    frac=(lambda n,d: (lambda l: [frac[0]*l//frac[1]+n*l//d,l])(lcm(frac[1],d)))(*a.internal[i][0])
                    del(a.internal[i])
                a.internal.append([frac,1])
            elif not cont:
                break
        a.internal.sort()
        if not a.internal:
            a.internal=[[[0,1],1]]
        return(a.internal)
    __eq__=(lambda a,b: a.internal==b.internal)
    __add__=(lambda a,b: surd(a.internal+b.internal) if type(b)==surd else a+surd(b))
    __radd__=__add__
    __neg__=(lambda a: surd(lap(lambda a: [[-a[0][0],a[0][1]],a[1]],a.internal)))
    __sub__=(lambda a,b: a+-b)
    __mul__=(lambda a,b: surd(map(lambda a: [[a[0][0]*b,a[0][1]],a[1]],a)) if type(b)==int else surd([(lambda l: [[sgn(n)*sgn(m)*abs(n**(l//e)*m**(l//f)),d**(l//e)*c**(l//f)],l])(lcm(e,f)) for ((n,d),e),((m,c),f) in product(a,b) if n and m]))
    __truediv__=(lambda a,b: surd(map(lambda t: [[t[0][0],t[0][1]*b],t[1]],a)) if type(b)==int else TypeError('wibble'))
    __float__=(lambda a: float(sum(map(lambda b: sgn(b[0][0])*(abs(b[0][0])/b[0][1])**(1/b[1]),a))))
    __bool__=(lambda a: any(map(lambda a: a[0][0],a)))
    __gt__=(lambda a,b: float(a)>float(b))
    def __init__(a,t):
        if type(t)==int:
            a.internal=[[[t,1],1]]
        else:
            a.internal=list(t)
            a.simplify()

permute=(lambda p,t: (lambda o: o+t[len(p):] if len(t)>len(p) else o)(tap(t.__getitem__,p)))

def floorctorial(n,i=False):
    k=1;a=1
    while a<n: k+=1;a*=k
    return(k-(a>n) if i else (a//k if a>n else a))

"""def ilog(n,b):
    min,max=0,1
    acc=b
    while acc<=n:
        acc**=2
        min=max
        max<<=1
    '''if acc==n:
        return(max)
    else:''' #indent all thereafter
    #if True:
    change=min>>1
    while change:
        max=acc//b**change
        if max<=n:
            acc=max
            min+=change
        change>>=1
    return(min)
    '''else:
        while max+~min:
            mid=max+min>>1
            if b**mid>n: max=mid
            else: min=mid
        return(min)'''""" #from OEIS, however I don't think bisection is more efficient when exponentiation takes time proportional to output length
def ilog(n,b):
    if b==1:
        return(inf)
    else:
        i=0
        while n>1:
            n//=b
            i+=1
        return(i-(not n))

def A000793(n,o=True): #highest lcm of integers summing to n
    V=lap(lambda _: 1,range(n+1))
    for i in primerange(n+1):
        for j in range(n,i-1,-1):
            '''hi=V[j]
            pp=i
            while pp<=j:
                hi=max((pp if j==pp else V[j-pp]*pp),hi)
                pp*=i
            V[j]=hi'''
            V[j]=reduce(lambda a,_: (max(a[0],a[1] if j==a[1] else V[j-a[1]]*a[1]),a[1]*i),range(ilog(j,i)),(V[j],i))[0]
        #V[i:n+1]=lap(lambda j: reduce(lambda a,_: (max(a[0],V[j-a[1]]*a[1]),a[1]*i),range(ilog(j-1,i)+1),(V[j],i))[0],range(i,n+1)) #cannot be done because they consider other values also
    return(V[-1] if o else V)
#print('\n'.join(map(str,enumerate(A000793(48,False)))))

#A003418=(lambda n: reduce(lcm,range(1,n+1),1))
A003418=(lambda n: reduce(int.__mul__,map(lambda p: p**ilog(n,p),primerange(n+1)),1)) #lcm of all length-n permutations' orders

class permutation:
    __repr__=(lambda p: 'permutation('+(','*(len(p)>9)).join(map(str,p.internal))+')')
    __iter__=(lambda p: iter(p.internal))
    __len__=(lambda p: len(p.internal))
    __getitem__=(lambda p,i: p.internal[i] if type(i)==slice or i<len(p) else i)
    __add__=(lambda a,b: permutation(a.internal+tap(lambda i: i+len(a),b.internal)))
    conjugate=(lambda p: permutation(map(p.index,range(len(p)))))
    #__pow__=(lambda p,n: p.conjugate() if n==-1 else reduce(lambda r,i: p*r,range(n-1),p) if n else range(len(p)))
    #__pow__=(lambda p,n: (lambda n: p.conjugate()**-n if n<0 else reduce(lambda r,i: p*r,range(n-1),p) if n else range(len(p)))(lambda o: ((lambda n: n-o*(n>o>>1))(n%o))(order(p))))
    def __pow__(p,n): #any asymptotically faster than this would require factorising n to enact recursively, probably
        c=p.cycles()
        '''m=tap(lambda c: (lambda e: c[-e:]+c[:-e])(n%len(c)),c)
        o=lap(lambda _: None,p) #do not multiply lists by integers (worst mistake of my life)
        for c,m in zip(c,m):
            for c,m in zip(c,m):
                o[c]=m'''
        o=lap(lambda _: None,p)
        for c in c:
            #for c,m in zip(c,(lambda e: c[-e:]+c[:-e])(n%len(c))):
            for c,m in zip(c,(lambda e: c[e:]+c[:e])((lambda o: (lambda n: n-o*(n>o>>1))(n%o))(len(c)))): #very elegant I think
                o[c]=m
        return(permutation(o))
    comp=(lambda a,b: (a,permutation(b.internal+tuple(range(len(a)-len(b))))) if len(a)>=len(b) else permutation.comp(b,a))
    __eq__=(lambda a,b: __eq__(*map(tuple,permutation.comp(a,b))))
    __gt__=(lambda a,b: __gt__(*permutation.comp(a,b)))
    __lt__=(lambda a,b: __lt__(*permutation.comp(a,b)))
    __ge__=(lambda a,b: __ge__(*permutation.comp(a,b)))
    __le__=(lambda a,b: __le__(*permutation.comp(a,b)))
    __mul__=(lambda a,b: permutation(permute(a,b)))
    __rmul__=__mul__
    index=(lambda p,i: p.internal.index(i))
    def __init__(p,t): #my feeling when it cannot be a lambda :-(
        p.internal=((lambda p: reduce(lambda t,i: ((len(p)+~t[1].pop(i),)+t[0],t[1]),p,((),list(range(len(p)))))[0])(shortduce(lambda t: (lambda m,d: (((m,)+t[0],d,t[2]+1),d))(*moddiv(t[1],t[2])),i=((),t,1))[0]) if type(t)==int else tuple(t)) #first version of int part was big-endian (assuming elements were prepended), (lambda p: reduce(lambda t,i: (t[0]+(t[1].pop(i),),t[1]),p,((),list(range(len(p)))))[0])(shortduce(lambda t: (lambda m,d: (((m,)+t[0],d,t[2]+1),d))(*moddiv(t[1],t[2])),i=((),t,1))[0])
    #__int__=(lambda p: sum(starmap(int.__mul__,enumerate(reversed(tuple(starmap(lambda i,t: t-sum(map(t.__gt__,p[:i])),enumerate(p)))),start=1))))
    #__int__=(lambda p: sum(starmap(int.__mul__,enumerate(starmap(lambda i,t: t-sum(map(t.__gt__,p[:i])),enumerate(p)),start=1))))
    #__int__=(lambda p: sum(starmap(int.__mul__,zip(reversed(tuple(starmap(lambda i,t: t-sum(map(t.__gt__,p[:i])),enumerate(p)))),redumulate(int.__mul__,range(len(p)),1)))))
    #__int__=(lambda p: sum(starmap(int.__mul__,zip(starmap(lambda i,t: t-sum(map(t.__gt__,p[:i])),enumerate(dbg(p))),redumulate(int.__mul__,range(1,len(p)))))))
    #__int__=(lambda p: sum(starmap(int.__mul__,zip(reversed(tuple(starmap(lambda i,t: t-sum(map(t.__gt__,p[:i])),enumerate(dbg(p))))),redumulate(int.__mul__,range(1,len(p)))))))
    #__int__=(lambda p: sum(starmap(int.__mul__,zip(starmap(lambda i,t: t-sum(map(t.__gt__,p[:i])),enumerate(dbg(p))),redumulate(int.__floordiv__,range(len(p),2),fact(len(p))))))) #produces sequence of palindromes, (0,2,0,12,6,12,0,24,0,72,24,72,0,48,0,72,48,72,24,48,24,72,48,72,0,120,0,240,120,240,0) (very suspicious)
    #__int__=(lambda p: sum(starmap(int.__mul__,zip(starmap(lambda i,t: t-sum(map(i.__gt__,map(p.index,range(i)))),enumerate(dbg(p))),redumulate(int.__floordiv__,range(len(p),2),fact(len(p))))))) #also palindromes
    #__int__=(lambda p: sum(starmap(int.__mul__,zip(starmap(lambda i,t: t+~sum(map(i.__lt__,map(p.index,range(i)))),enumerate(dbg(p))),redumulate(int.__floordiv__,range(len(p),1),fact(len(p))))))) #very suspicious non-palindromic sequence of alternating sign
    #__int__=(lambda p: sum(starmap(int.__mul__,zip(starmap(lambda i,t: t+1-sum(map(t.__gt__,p[i+1:])),enumerate(dbg(p))),redumulate(int.__floordiv__,range(len(p),2),fact(len(p))))))) #A048765
    #__int__=(lambda p: sum(starmap(int.__mul__,zip(starmap(lambda i,t: t+1-sum(map(t.__gt__,p[:i])),enumerate(dbg(p))),redumulate(int.__floordiv__,range(len(p),2),fact(len(p))))))) #weird Thue-Morse-like thing
    #__int__=(lambda p: sum(starmap(int.__mul__,zip(starmap(lambda i,t: t-sum(map(i.__lt__,map(p.index,range(i)))),enumerate(dbg(p))),redumulate(int.__floordiv__,range(len(p)-1,0,-1),fact(len(p)-1))))))
    __int__=(lambda p: reduce(lambda r,i: (r[0]+(len(p)+~(i[1]+sum(map(i[1].__lt__,r[1]))))*fact(len(p)+~i[0]),r[1]+(i[1],)),enumerate(p[:0:-1]),(0,()))[0])

    def cycles(p,o=0): #len(cycles) if o==2 else (oscillatory period) if o else (cycles themselves) (idea to use sets is from https://stackoverflow.com/a/75823973 :-)
        pi={i: p for i,p in enumerate(p)}
        cycles=(o!=2 if o else [])
        while pi:
            nexter=pi[next(iter(pi))] #arbitrary starting element
            cycle=(not(o) and [])
            while nexter in pi:
                if o: cycle+=1;curr=nexter;nexter=pi[nexter];del(pi[curr]) #a little bit of tessellation (very suspicious)
                else: cycle.append(nexter);nexter=pi[nexter];del(pi[cycle[-1]])
            if o==2: cycles+=1
            elif o: cycles=lcm(cycles,cycle)
            else: cycles.append(cycle) #inelegant (however I am not quite deranged enough for eval('cycles'+('+=1' if o==2 else '=lcm(cycles,cycle)' if o else '.append(cycle)')) :-)
        return(cycles)
    order=(lambda p: p.cycles(1))
    maxder=(lambda p: A000793(len(p)))
    modder=(lambda p: A003418(len(p))) #could be used instead of order, perhaps (depending)
    #parity=(lambda p: reduce(int.__xor__,((a<b)&(B<A) for (a,A),(b,B) in product(enumerate(p),repeat=2)))) #very suspicious (from https://codegolf.stackexchange.com/questions/75841/75856)
    #parity=(lambda p,b=None: reduce(lambda c,d: c^~len(d)&1,permutation.cycles(p),0) if b==None else permutation.parity(tap(p.index,b))) #O(n*lg(n)) (lg due to hashtables) instead of O(n**2) #may be computing that of inverse but parity is irrespective
    parity=(lambda p,b=None: len(p)-p.cycles(2)&1 if b==None else permutation.parity(tap(p.index,b))) #O(n*lg(n)) (lg due to hashtables) instead of O(n**2) #may be computing that of inverse but parity is irrespective

#print(','.join(map(lambda n: str(cycles(permutation(n),True)),range(16))))
'''n=0
a=b=()
while True:
    b+=tap(lambda k: int(permutation(k)),((lambda f: range(f,f*n))(fact(n-1)) if n else (0,)))
    print(b)
    #print(n,rle(sorted(map(lambda k: permutation(k).cycles(True),((lambda f: range(f,f*n))(fact(n-1)) if n else (0,))))))
    a=tarmap(int.__add__,zip_longest(a,(lambda s: tap(s.count,range(A000793(n)+1)))(sorted(map(lambda k: permutation(k).cycles(True),((lambda f: range(f,f*n))(fact(n-1)) if n else (0,))))),fillvalue=0)) #A057731
    #print(str(n)+':',','.join(map(str,a))+',')
    #print(str(n)+':',','.join(map(lambda r: str(r[1]),rle(sorted(map(lambda k: permutation(k).cycles(True),range(fact(n)))))))+',') #rle version of above sequence, so forgoing 0s (not very interesting)
    #print(str(n)+':',str(max(map(lambda k: permutation(k).cycles(True),((lambda f: range(f,f*n))(fact(n-1)) if n else (0,))),default=0))+',') #A000793
    #print(n,lcm(*map(lambda k: permutation(k).cycles(True),((lambda f: range(f,f*n))(fact(n-1)) if n else (0,)))))
    n+=1
    if n>5:
        break
exit()'''

'''print('\n'.join(map(lambda n: str(permutation(n)),range(16))))
print(tap(lambda n: permutation(n).order(),range(16)))
print(tap(lambda n: A002262(n,True),range(16)))
#print(tap(permutation,range(16)))
print(tap(lambda n: int(permutation.__mul__(*map(permutation,A002262(n,True)))),range(16)))
print(tap(lambda n: int(permutation.__mul__(*map(permutation,reversed(A002262(n,True))))),range(16)))
print(tap(lambda n: int(permutation(n)**-1),range(16))) #A056019

import matplotlib.pyplot as plot
from numpy import array
r=fact(8)
mode=3
if mode==3:
    plot.scatter(range(r),tap(lambda n: int(permutation(n)**-1),range(r)),s=1) #change -1 to n for some more interestingness
elif mode==2:
    plot.imshow(array(tap(lambda n: tap(lambda k: int(permutation(n)*permutation(k)),range(r)),range(r))))#plot.scatter(range(r),tap(lambda n: int(permutation.__mul__(*map(permutation,A002262(n,True)))),range(r)),s=1);plot.show()
elif mode:
    plot.scatter(range(r),tap(lambda n: int(permutation.__mul__(*map(permutation,reversed(A002262(n,True))))),range(r)),s=1)
else:
    plot.scatter(range(r),tap(lambda n: int(permutation.__mul__(*map(permutation,A002262(n,True)))),range(r)),s=1)
plot.show()
exit()''' #some interesting plots

eventations=(lambda v: tap(lambda p: tap(v.__getitem__,p),filter(lambda p: not(permutation(p).parity()),permutations(range(len(v))))))
signventations=(lambda v,t=None: tap((vector3 if len(v)==3 else versor) if t==None else t,chap(signs,eventations(v))))

signs=(lambda q,alge=True: (lambda e: tap(lambda n: reduce(lambda c,q: (c[0]+(q*(-1)**(n>>c[1]&1),),c[1]+1) if q else (c[0]+(surd(0) if alge else 0,),c[1]),q,((),0))[0],range(1<<len(e))))(tuple(filter(lambda x: x[1],enumerate(q)))))
orthoplex=(lambda d: tap(lambda n: (vector3 if d==3 else versor)((n>>1)*(surd(0),)+(surd((-1)**n),)+(d+~(n>>1))*(surd(0),)),range(2*d)))
cell600=orthoplex(4)+tap(versor,signs((surd([[[1,2],1]]),)*4))+signventations((surd([[[1,4],1],[[5,16],2]]),surd([[[-1,4],1],[[5,16],2]]),surd([[[1,2],1]]),surd(0)))
icosahedronDisplacement=(1/sqrt(5),2/sqrt(5)) #(sqrt((1+1/sqrt(5))/2),sqrt((1-1/sqrt(5))/2))
fifths=tuple(zip((lambda t: t+t[:0:-1])((1,(sqrt(5)-1)/4,(-sqrt(5)-1)/4)),(lambda t: t+tap(float.__neg__,t[:0:-1]))((0,sqrt((5+sqrt(5))/2)/2,sqrt((5-sqrt(5))/2)/2))))
polarRotate=(lambda t,r: (cos(r),cos(t)*sin(r),sin(t)*sin(r))) #3D spherical coordinates, parameters (theta,radius)
polarMultiply=(lambda t,r: vector3((r[0],t[0]*r[1],t[1]*r[1])))
#icosahedron=tap(lambda i: vector3(map(surd,((-1)**i,0,0))),range(2))+(lambda t: t+tap(lambda t: vector3(map(float.__neg__,t)),t))(tap(lambda f: polarMultiply(f,icosahedronDisplacement),fifths))
icosahedron=(lambda t: t+tap(vector3.__neg__,t))(tap(vector3,\
((1,          0,             0),
 (1/sqrt(5),  2/sqrt(5),     0))+(lambda t: t+tap(lambda t: (t[0],t[1],-t[2]),t))(
((1/sqrt(5),(-1/sqrt(5)+1)/2,sqrt((1+1/sqrt(5))/2)),
 (1/sqrt(5),(-1/sqrt(5)-1)/2,sqrt((1-1/sqrt(5))/2))))))

icosahedronRotations=tap(lambda q: versor((0,0,0,1))*versor(q+(0,)), #the multiplication by k is to rotate by 180º to keep relative directions to adjacent vertices aligned (precede with (lambda t: t+tap(versor.conjugate,t)) for infinite set)
(((sqrt((1+1/sqrt(5))/2), sqrt((1  -1/sqrt(5))/2),   0),)+(lambda t: t+tap(lambda t: (t[0],t[1],-t[2]),t))(
((sqrt((1+1/sqrt(5))/2), sqrt((1/2-1/sqrt(5))/2),   1/2),
 (sqrt((1+1/sqrt(5))/2),-sqrt((1  +1/sqrt(5))/2)/2,(1-sqrt(5))/4))))) #for quaternion rotations (which seem to be doubled)

#I didn't foresee nested roots when making the surd class so it computes them numerically with epsilons (these were all found manually by... uhhh ...deterministic methods (and not RIES :-))
#      surd forms                                      (phi versions (if you dislike 5),   sqrts of quadratic roots)
(a,b)=( sqrt((1-1/sqrt(5))/2), sqrt((1+1/sqrt(5))/2)) #(1/sqrt(2+phi),      sqrt(phi/(2*phi-1)))   #x**2-  x  +1/ 5
(c,d)=( sqrt(1-2/sqrt(5))  /2, sqrt(1+2/sqrt(5))  /2) #(sqrt(1/(2+phi)-1/4),sqrt(1/(2-1/phi)-1/4)) #x**2-  x/2+1/80
(e,f)=((sqrt(5)-1)         /4,(sqrt(5)+1)         /4) #(1/(2*phi),          phi/2)                 #x**2-3*x/4+1/16 (roots of x**2-sqrt(5)*x/2+1/4)
(g,h)=( sqrt((5-sqrt(5))/2)/2, sqrt((5+sqrt(5))/2)/2) #(sqrt(2-1/phi)/2,    sqrt(2+phi)/2)         #x**2-5*x/4+5/16
(i,j)=( a                  /2, b                  /2)                                              #x**2-  x/4+1/80
k=1/2
swap=(lambda q: (q,(q[1], q[0], q[3],-q[2])))
negr=(lambda q: (q,(q[0], q[1],-q[2],-q[3])))
conj=(lambda q: (q,(q[0],-q[1],-q[2],-q[3])))
icosahedronOrientations=tap(lambda q: (lambda q: (q,versor.canonicalise(q.conjugate())))(versor(q)),chap(chain.from_iterable,(tap(swap,((1,0,0,0),(0,0,a,b))),tap(lambda q: chain.from_iterable(tap(negr,swap(q))),((e,0,0,h),(f,0,0,g),(0,k,c,b),(e,0,b,j),(f,0,a,i),(0,k,d,a))),tap(lambda q: chap(conj,chap(negr,swap(q))),((e,f,i,j),(f,k,c,i),(k,e,j,d),(k,k,d,c))))))

icosidodecahedron=orthoplex(3)+signventations((surd([[[-1,4],1],[[5,16],2]]),surd([[[1,2],1]]),surd([[[1,4],1],[[5,16],2]])))

#__import__('pprint').pprint(cell600)
#__import__('pprint').pprint(icosahedron);exit()
#print(len(cell600))

#print(fratrix(transitions,hidezero=False))
#print('\\\n'.join(map(lambda t: ''.join(map(lambda t: (lambda x: '0'*(2-len(x))+x)(hex(t)[2:]),t)),transitions))) #very suspicious

edgeActions=tap(lambda q: cell600.index(versor((surd([[[1,4],1],[[5,16],2]]),)+q)),chain(*map(signs,eventations((surd([[[-1,4],1],[[5,16],2]]),surd([[[1,2],1]]),surd([[[0,1],1]])))))) #((1+sqrt(5))/4,(sqrt(5)-1)/4,1/2,0)
lineDrawers=tap(edgeActions.__getitem__,(0,1,4,5,8,9))#pairs=((0,3),(1,2),(4,7),(5,6),(8,11),(9,10))

def group(group,identity=None): #mainly for permutations
    if identity==None:
        identity=permutation(range(len(group[0]))) #assume acting upon 
    iterations=0
    changes=0
    while not iterations or changes:
        length=len(group)
        exponents=[]
        for i,t in enumerate(group):
            exponents.append([])
            order=identity
            j=0
            while True: #len(cell600)+1
                order*=t
                exponents[-1].append(order)
                if j and order==identity:
                    break
                j+=1
        group=tuple(sorted(sap(lambda x: tuple(reduce(__mul__,x,identity)),product(chain.from_iterable(exponents),repeat=2)))) #unfortunately computers exist in polynomially large space so we cannot be elegant with product(*exponents) :-(
        changes=len(group)-length
        iterations+=1
    return(group)

def coverers(transitions):
    exponents=[]
    transitionPeriods=[]
    roots=lap(lambda _: -1,transitions)
    for i,t in enumerate(transitions):
        periods=lap(lambda _: 0,transitions)
        order=tuple(range(len(transitions)))
        exponents.append([])
        j=0
        while True: #len(cell600)+1
            if j and not transitions.index(order):
                break
            exponents[-1].append(transitions.index(order))
            j+=1
            order=permute(order,t)
            if roots[exponents[-1][-1]]==-1:
                roots[exponents[-1][-1]]=i
            for k,(o,p) in enumerate(zip(order,periods)):
                if not p and k==o:
                    periods[k]=j
        #print(i,exponents[-1])
        transitionPeriods.append(periods[0])
    #print(rle(sorted(transitionPeriods)))
    '''colourdef='lambda c: ('+'+'.join(map(lambda r: str(r[1])+'*c**'+str(len(transitions)//r[0]),rle(sorted(transitionPeriods))))+')//'+str(len(transitions))
    colourer=eval(colourdef) #ie. number of c-colourings of a 120-cell's faces under action of icosians (moving each face to (1,0,0,0) (without rotating it perpendicularly)) being considered equivalent
    print(colourdef)
    print(tap(colourer,range(16)))'''
    '''for i in range(len(roots)):
        for j,t in enumerate(roots):
            if roots[t]!=t:
                roots[j]=roots[t]'''
    factors=lap(lambda _: 0,exponents)
    for i,e in enumerate(exponents):
        for x in e[:-1]:
            if x!=i:
                factors[x]+=1
    primes=tuple(filter(lambda i: not(factors[i]),range(len(exponents))))
    primexps=tap(exponents.__getitem__,primes)
    i=0
    while True:
        i+=1
        #print(i)
        for p in combinations(primexps,i):
            if len(tuple(sap(lambda x: reduce(lambda a,b: transitions[a][b],x),product(*p))))==len(transitions):
                return(tap(lambda e: e[1],p))
                #print(p,x)
                #coveringActions=p
                #break
        #if len(x)==len(transitions): break
multiplicationTable=(lambda s: tap(lambda c: tap(lambda d: s.index(c*d),s),group(s)))
#coveringActions=(2,4,8,72,74)#=coverers(multiplicationTable(cell600))
#print(tap(cell600.__getitem__,coverers(multiplicationTable(cell600))));exit()

generators=\
(( 2,  3,  1,  0,  6,  7,  5,  4, 13, 15, 12, 14, 21, 23, 20, 22,  9, 11,  8, 10, 17, 19, 16, 18, 49, 51, 48, 50, 53, 55, 52, 54, 77, 79, 76, 78, 73, 75, 72, 74, 98, 99,102,103, 96, 97,100,101, 29, 31, 28, 30, 25, 27, 24, 26, 81, 83, 80, 82, 85, 87, 84, 86,106,107,110,111,104,105,108,109, 33, 35, 32, 34, 37, 39, 36, 38, 61, 63, 60, 62, 57, 59, 56, 58,114,115,118,119,112,113,116,117, 43, 42, 47, 46, 41, 40, 45, 44, 67, 66, 71, 70, 65, 64, 69, 68, 91, 90, 95, 94, 89, 88, 93, 92),
 ( 4,  5,  7,  6,  1,  0,  2,  3, 17, 21,  9, 13, 16, 20,  8, 12, 19, 23, 11, 15, 18, 22, 10, 14, 93, 95, 89, 91, 92, 94, 88, 90,108,110,104,106,109,111,105,107, 57, 61, 56, 60, 59, 63, 58, 62,116,118,112,114,117,119,113,115, 45, 47, 41, 43, 44, 46, 40, 42, 73, 77, 72, 76, 75, 79, 74, 78, 69, 71, 65, 67, 68, 70, 64, 66,100,102, 96, 98,101,103, 97, 99, 25, 29, 24, 28, 27, 31, 26, 30, 85, 81, 84, 80, 87, 83, 86, 82, 37, 33, 36, 32, 39, 35, 38, 34, 53, 49, 52, 48, 55, 51, 54, 50),
 ( 8, 23, 17, 14, 11, 20, 13, 18,  9,  1,  6, 15,  2, 21, 12,  5,  4, 19, 10,  3, 16,  7,  0, 22,104, 39, 64, 79, 72, 71, 32,111, 96, 47, 80, 63, 56, 87, 40,103,112, 31, 48, 95, 88, 55, 24,119, 73, 37, 66,107,108, 69, 34, 78, 89, 27,113, 51, 52,118, 28, 94, 81, 43, 98, 61, 58,101, 44, 86, 57, 45, 82, 99,100, 85, 42, 62, 49, 29,114, 91, 92,117, 26, 54, 65, 35, 76,110,105, 75, 36, 70, 25,115, 53, 90, 93, 50,116, 30, 41, 83,102, 60, 59, 97, 84, 46, 33, 67, 77,106,109, 74, 68, 38),
 (72, 79, 37, 34, 65, 70,106,109, 49, 43, 80,115,102, 95, 28, 62, 57, 27, 88, 97,116, 87, 44, 54, 96, 35,  8, 51, 52, 23, 36,103,  2, 31, 32, 99,100, 39, 24,  3, 48, 15, 76, 63, 56, 75, 16, 55, 77, 29, 42, 14, 17, 45, 26, 74,  9, 41, 64,113,118, 71, 46, 22,114, 67, 60,  5,  4, 59, 68,117, 73,  1, 40, 50, 53, 47,  0, 78, 98, 91, 12,107,108, 19, 92,101,112, 83, 20,111,104, 11, 84,119, 33, 82, 21, 30, 25, 10, 85, 38, 81,  6,110, 94, 89,105,  7, 86, 13, 66, 61, 90, 93, 58, 69, 18),
 (74, 77, 36, 35, 69, 66,104,111, 92,100, 59, 25, 40, 48,113, 81, 86,118, 55, 47, 30, 60, 99, 91, 18, 53,101, 37, 34, 98, 50, 13, 26,  2, 97, 33, 38,102,  3, 29, 58, 73, 10, 49, 54, 21, 78, 61, 24, 72, 11, 41, 46, 20, 79, 31, 68,116, 19, 45, 42, 12,115, 67,  4, 57, 64,112,119, 71, 62,  5, 44, 52, 75,  1,  0, 76, 51, 43, 88, 96,105,  9, 22,110,103, 95,109, 17, 82,114,117, 85, 14,106, 16, 27, 32, 83, 84, 39, 28, 15,108, 93, 80,  6,  7, 87, 90,107, 56, 89,  8, 65, 70, 23, 94, 63))
genxponents=[]
for i,t in enumerate(generators):
    genxponents.append([])
    order=tuple(range(len(t)))
    j=0
    while True:#len(cell600)+1
        order=permute(order,t)
        genxponents[-1].append(order)
        if j and order==tuple(range(120)):
            break
        j+=1
transitions=tuple(sorted(sap(lambda x: reduce(permute,x),product(*genxponents))))
#print(fratrix(transitions));exit()

import pygame
from pygame.locals import *
clock=pygame.time.Clock()
pygame.init()
size=(2560,1050)
screen=pygame.display.set_mode(size[:2],pygame.RESIZABLE)
mouse=pygame.mouse
def drawShape(size,pos,colour,shape=0):
    size=min(size,minSize)
    '''if shape==0:
        pygame.draw.rect(screen,colour,pos+size)
    elif shape<5:
        pygame.draw.polygon(screen,colour,[[p+s/2*cos(((i+shape/2)/(2 if shape==4 else 3 if shape==3 else  4)+di/2)*pi) for di,(p,s) in enumerate(zip(pos,size))] for i in range(4 if shape==4 else 6 if shape==3 else 8)])
    else:'''
    pygame.draw.circle(screen,colour,pos,size/2)
def drawLine(initial,destination,colour,width=1):
    if width==1:
        pygame.draw.line(screen,colour,initial,destination)
    else:
        (x,y)=map(lambda d,i: (d-i)**2,destination,initial)
        if x or y:
            pygame.draw.line(screen,colour,initial,destination,int(width*((x/y if x<y else y/x)+1)**0.5))
def doEvents():
    global run,clickDone
    clickDone=False
    global framesMouseDown,shortClickDone
    global size,minSize,halfSize
    shortClickDone=False
    if mouse.get_pressed()[0]:
        framesMouseDown+=1
    else:
        framesMouseDown=0
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            run=False
        if event.type==pygame.MOUSEBUTTONUP:
            clickDone=True
            if framesMouseDown<0.2*FPS:
                shortClickDone=True
        if event.type==pygame.WINDOWRESIZED:
            size=screen.get_rect().size
            minSize=min(size)
            halfSize=tap(lambda s: s/2,size)
    pygame.display.flip()
    clock.tick(FPS)
    screen.fill(0x000000)
    size=screen.get_size()
FPS=24
axialCollision=(lambda m0,v0,m1,v1: (((m0-m1)*v0+2*m1*v1)/(m0+m1),
                                ((m1-m0)*v1+2*m0*v0)/(m1+m0)))

shapes=(cell600,icosahedron,icosidodecahedron)
shapedims=(4,3,3)
shape=0
dims=shapedims[shape]
surdShape=shapes[shape]
nodes=tap(lambda q: type(q)(map(float,q)),surdShape)

pixelAngle=tau/max(size)
gain=1/6000
rad=10
drag=1/2**4
angularGain=gain*tau/FPS
gain=angularGain
vertexPointing=False #point towards vertex such that another is directly to its right
(c,s)=(sqrt((1/2+1/sqrt(5))/2),sqrt(1-2/sqrt(5))/2)
camera=[((versor((2/sqrt(5),0.0,-1/2,0.0)),versor((2/sqrt(5),0.0,1/2,0.0))) if False else (versor((az,-b,-b,az)),versor((az,b,b,-az)))) if vertexPointing else (versor((1.0,0.0,0.0,0.0)) if dims==4 else vector3((0.0,0.0,0.0)),versor((1.0,0.0,0.0,0.0))),[versor((1.0,0.0,0.0,0.0)) if dims==4 else vector3((0.0,0.0,0.0)),versor((1.0,0.0,0.0,0.0))]]#((versor((1.0,0.0,0.0,0.0)),versor((1.0,0.0,0.0,0.0))),(versor((1.0,0.0,0.0,0.0)),versor((1.0,0.0,0.0,0.0)))) #velocity is versor by which it's multiplied each frame
versor((1.0,0.0,0.0,0.0))
oldCamera=camera[0]

if dims==4:
    #print(tap(lambda q: hypot(*q),icosahedronRotations));exit()
    interpolate=2
    interpolationFactors=tap(lambda l: nodes[l]**(1/interpolate),lineDrawers)
    if shape==0:
        connections=tap(lambda t: tap(lambda e: t[e],edgeActions),transitions)
    '''for i in range(13):
        print(camera[0])
        camera[0]=(lambda e,r: (r*e[0],e[1]*r**-1))(camera[0],icosahedronRotations[1])'''
    epsilon=1/2**16 #don't want to be too strict
    '''manyEyes=[camera[0]] #(only darkness now :::-)
    stalks=[[-1]*5] #would be connections but name already taken (more surreal this way)
    additiondex=0
    oldAdditions=0
    while True:
        additions=0
        stalksConnected=0
        moves=tap(lambda e: tap(lambda r: tap(versor.canonicalise,(r*e[0],e[1]*r**-1)),icosahedronRotations),manyEyes[additiondex:])
        for i,n in enumerate(moves):
            for j,m in enumerate(n):
                if m==(versor((0.0,0.0,0.0,0.0)),versor((0.0,0.0,0.0,0.0))):
                    print('\n\n'.join(map(str,('wibble',n,icosahedronRotations[j],m,(icosahedronRotations[j]*manyEyes[additiondex+i][0],manyEyes[additiondex+i][1]*icosahedronRotations[j]**-1),tap(versor.canonicalise,(icosahedronRotations[j]*manyEyes[additiondex+i][0],manyEyes[additiondex+i][1]*icosahedronRotations[j]**-1)),sgn(icosahedronRotations[j]*manyEyes[additiondex+i][0])))));exit()
                for k,e in enumerate(manyEyes):
                    if hypot(*map(lambda a,b: hypot(*(a-b)),m,e))<epsilon: #vector3.abs(versor.log(m/e)) #dist(*map(chain.from_iterable,(m,e)))
                        #print(m,e)
                        #print(dist(*map(chain.from_iterable,(m,e))))
                        #print(len(stalks),additiondex,i,k,additions,len(moves))
                        stalks[additiondex+i][j]=k
                        stalksConnected+=1
                        break
                else:
                    print(m)
                    manyEyes.append(m)
                    stalks.append([-1]*5)
                    #print('l',len(stalks),'a',additions,'d',additiondex,'i',i,'k',k,'j',j)
                    additions+=1
                    stalks[additiondex+i][j]=additiondex+oldAdditions+additions
        additiondex+=oldAdditions
        oldAdditions=additions
        #print('wibble',additions,len(manyEyes),stalksConnected)
        if not additions:
            break
    #print('\n'.join(map(lambda s: '('+','.join(map(str,s))+')',zip(*stalks))))
    print(fratrix(stalks))#tuple(zip(*stalks))
    __import__('pprint').pprint(manyEyes)
    print(len(manyEyes),len(stalks))''' #this part was used to generate icosahedronOrientations group-style (then I sorted it by conjugacy class for the current one)
    icosahedralGroup=tap(permutation,transpose(tap(lambda e: tap(lambda r: (lambda m: next(filter(lambda e: hypot(*map(lambda a,b: hypot(*(a-b)),m,icosahedronOrientations[e]))<epsilon,range(len(icosahedronOrientations)))))(tap(versor.canonicalise,(r*e[0],e[1]*r**-1))),icosahedronRotations),icosahedronOrientations)))
    print(fratrix(icosahedralGroup))
    icosahedronCoverers=coverers(group(icosahedralGroup))
    #exit()
else: #duals
    interpolate=0
    threshold=(4/3 if shape==1 else 2/3) #very suspicious (1+sqrt(5)>=3)
    connections=tarmap(lambda i,n: tap(lambda x: x[0],filter(lambda m: vector3.abs(m[1]-n)<threshold,enumerate(nodes[:i]))),enumerate(nodes))
    biconnections=tarmap(lambda i,n: tap(lambda c: c[0],filter(lambda m: m[0] in n or i in m[1],enumerate(connections))),enumerate(connections))
    dualMode=False
    if dualMode:
        snakes=[]
        for l in (3,5): #icosidodecahedron-specific
            snakes.append([])
            for b in combinations(range(len(nodes)),l):
                '''for c,d in combinations(b,2):
                    if c[0] in d[1] or d[0] in c[1]:'''
                snake=[b[0]]
                for j in range(l-1):
                    paths=tuple(filter(lambda c: c in b and c not in snake,biconnections[snake[-1]]))
                    '''if len(paths)!=2:
                        break'''
                    if paths:
                        snake.append(paths[0])
                    else:
                        break
                else:
                    if snake[0] in biconnections[snake[-1]]:
                        snakes[-1].append(snake)
        print(snakes)
        dual=tap(lambda s: tap(lambda s: sum(map(surdShape.__getitem__,s),start=vector3((0,0,0))),s),snakes) #I hate sum so much
        print(dual)
        suspicious=(shape==2)
        surdShape=(dual[1] if suspicious else chain.from_iterable(dual))
        nodes=tap(lambda q: (lambda q: 1/vector3.abs(q)*q)(type(q)(map(float,q))),surdShape)
        if suspicious:
            threshold*=2
        connections=(tarmap(lambda i,n: tap(lambda x: x[0],filter(lambda m: vector3.abs(m[1]-n)<threshold,enumerate(nodes[:i]))),enumerate(nodes)) if suspicious else (lambda s: tap(lambda n: tap(lambda x: x[0],filter(lambda m: len(tuple(filter(n.__contains__,m[1])))==2,enumerate(s))),s))(tuple(chain.from_iterable(snakes))))

project=(lambda n: camera[0][1]*(n+camera[0][0]) if dims==3 else camera[0][1]*versor.log(n/camera[0][0]) if twistyMode else versor.log(camera[0][0]*n*camera[0][1]))
def toScreen(position,radius=rad):
    (x,y,z)=position
    if projectionMode==0:
        r=tap(__add__,(position*minSize*(1/2))[:2],halfSize)
    if projectionMode==1: #weird
        r=tap(lambda i: minSize*atan2(i,z),(x,y))+(0,)
    elif projectionMode==2: #azimuthal equidistant
        h=((x or y) and atan2((x**2+y**2),z**2)/hypot(x,y))
        r=(x*h/pixelAngle,y*h/pixelAngle)
    else:
        magnitude=atan2(hypot(x,y),z) #other azimuthals
        if projectionMode==3: #Lambert equi-area (not to be taken to before 1772)
            magnitude=2*abs(sin(magnitude/2)) #formerly sqrt(sin(magnitude)**2+(cos(magnitude)-1)**2)
        elif projectionMode==4: #stereographic (trippy)
            if radius==0:
                magnitude=1/tan(magnitude/2) #equivalent to sin(magnitude)/(1-cos(magnitude))
            else:
                h=hypot(x,y)
                hh=hypot(*position)
                offset=asin(radius/hh)
                s0=1/tan((magnitude-offset)/2)
                s1=1/tan((magnitude+offset)/2)
                radius=s1-s0
                magnitude=(s0+s1)/2
        direction=atan2(x,y)
        r=(sin(direction)*magnitude/pixelAngle,cos(direction)*magnitude/pixelAngle)
    return((r[0]*zoom+halfSize[0],r[1]*zoom+halfSize[1],hypot(*position)))

projectionMode=3
perspectiveMode=True
twistyMode=False or not perspectiveMode #left because I think it is elegant but unfortunately misunderstood maths (curiously, motion seems to impart rotation)
omnidirectionalMode=(dims==3)
around=False
colourMode=(dims==4 or not(perspectiveMode))
triangleWave=(lambda o: tap(lambda c: int(255*(1-(lambda o: 3*o if o<1/3 else 2-3*o if o<2/3 else 0)((o+c/3)%1))),range(3)))
toggleKeys=(pygame.K_SPACE,pygame.K_x,pygame.K_y,pygame.K_z)
oldToggles=(False,)*len(toggleKeys)
debugMode=False
run=True
while run:
    keys=pygame.key.get_pressed()
    doEvents()
    zoom=hypot(*size)/minSize
    if keys[pygame.K_LSHIFT]:
        camera=[camera[0],[versor((1.0,0.0,0.0,0.0)) if dims==4 else vector3((0.0,0.0,0.0)),versor((1.0,0.0,0.0,0.0))]]
    elif twistyMode: #camera=((position,orientation),(spatial,angular))
        camera=tarmap(lambda s,v: tap(versor.__mul__,s,v),(camera,(camera[1],map(lambda i,v,m: vector3.exp((lambda m: tuple(camera[0][1].conjugate()*m) if i else m)(m*v)),range(2),(vector3((keys[pygame.K_d]-keys[pygame.K_a],keys[pygame.K_f]-keys[pygame.K_r],keys[pygame.K_w]-keys[pygame.K_s])),vector3((keys[pygame.K_DOWN]-keys[pygame.K_UP],keys[pygame.K_LEFT]-keys[pygame.K_RIGHT],keys[pygame.K_q]-keys[pygame.K_e]))),(gain,angularGain)))))
    else: #camera=((weird intertwined representation of both),(spatial,angular))
        spatial=gain*vector3((keys[pygame.K_a]-keys[pygame.K_d],keys[pygame.K_r]-keys[pygame.K_f],keys[pygame.K_s]-keys[pygame.K_w]))
        if dims==4:
            camera[1][0]*=vector3.exp(spatial)
        else:
            camera[1][0]+=camera[0][1].conjugate()*spatial
        camera[1][1]*=vector3.exp(angularGain*vector3((keys[pygame.K_DOWN]-keys[pygame.K_UP],keys[pygame.K_LEFT]-keys[pygame.K_RIGHT],keys[pygame.K_q]-keys[pygame.K_e])))
        if dims==4:
            motion=versor.log(camera[1][0])
            a=hypot(*motion)
            (left,right)=rotationParameters(a,(1.0,0.0,0.0,0.0),vector3.exp(pi/2*sgn(motion)))
            camera[0]=(camera[1][1]*left*camera[0][0],camera[0][1]*right*camera[1][1]**-1) #very important order
        else:
            camera[0]=(camera[0][0]+camera[1][0],camera[1][1]*camera[0][1])
    rotatedNodes=[];screenNodes=[];nodeColours=lap(lambda n: 0x000000,nodes);arounds=lap(lambda n: False,nodes)
    for i,n in enumerate(nodes):
        if perspectiveMode:
            f=project(n)
        else:
            f=n/camera[0][0]
            if colourMode:
                a=atan2(f[2],f[3])/tau
        arounds[i]=(f[2]<=0)
        '''if arounds[i]:
            f+=tau*sgn(f)'''
        o=toScreen(f)
        #print(str(f)+'\n'+str(o))
        nodeColours[i]=(0xbfbfbf if arounds[i] or f[2]<0 else 0xffffff) #(triangleWave(o[2]/pi) if debugMode else ((int(255*abs(cos((o[2]-pi)/2)))&0xf if abs(o[2])>pi else 255)*0x010101 if perspectiveMode else triangleWave(a)) if dims==4 and colourMode else 0xffffff)
        if omnidirectionalMode or not arounds[i] and f[2]>0:
            drawShape(rad/(o[2] if dims==3 else sin(o[2])),o[:2],nodeColours[i],0)
        rotatedNodes.append(f);screenNodes.append(o)
    for c,t,r,a,n,s in zip(connections,transitions,rotatedNodes,arounds,nodes,screenNodes): #very meta
        if interpolate:
            b=n;g=s
            for e,i in zip(lineDrawers,interpolationFactors):
                d=toScreen(project(nodes[e]))
                if dist(o[:2],d[:2])>20 and (omnidirectionalMode or d[2]>0 and g[2]>0): #and dist(r,rotatedNodes[t[e]])<1: #and not(a^arounds[t[e]]):
                    #print(e,t[e],z,t)
                    #print(r,rotatedNodes[t[e]])
                    for j in range(interpolate):
                        c=b*i
                        h=toScreen(project(c))
                        drawLine(g[:2],h[:2],0xffffff)
                        b=c;g=h
        else:
            for e in (map(t.__getitem__,lineDrawers) if dims==4 else c):
                g=project(nodes[e])
                if colourMode:
                    b=atan2(g[2],g[3])/tau
                d=toScreen(g)
                if dist(s[:2],d[:2])>20 and (omnidirectionalMode or d[2]>0 and g[2]>0):
                    drawLine(s[:2],d[:2],(tap(lambda a,b: sqrt((a**2+b**2)/2),triangleWave(a),triangleWave(b)) if colourMode else 0xffffff))
    '''for n in map(lambda n: camera[0][0]*n,versor((cos(pi/8),sin(pi/8),0.0,0.0))):
        f=camera[0][1]*versor.log(n/camera[0][0])
        drawShape(rad/o[2],o[:2],(triangleWave(a) if colourMode else 0xffffff),0)'''
    toggles=tap(keys.__getitem__,toggleKeys)
    for i,(k,o) in enumerate(zip(toggles,oldToggles)):
        if not k and o:
            if i==0: #space
                if True:
                    print('o',oldCamera)
                    print('c',camera[0])
                    print('d',tap(lambda o,c: o**-1*c,oldCamera,camera[0]))
                    oldCamera=camera[0]
                else:
                    #l=vector3((0,0,pi/5))#2*versor.log(sqrt((5-sqrt(5))/2)/2)#(nodes[edgeActions[3]])
                    (left,right)=rotationParameters(pi/5,(1.0,0.0,0.0,0.0),vector3.exp(pi/2*vector3((0,0,1))))
                    camera[0]=(camera[1][1]*left*camera[0][0],camera[0][1]*right*camera[1][1]**-1)
            elif i<4:
                angle=( (pi/2-atan(1/2))/2 #sqrt((1+1/sqrt(5))/2),sqrt((1-1/sqrt(5))/2)
                       if i<3 else #keys[pygame.K_LSHIFT]
                        pi/10)*(-1)**keys[pygame.K_RSHIFT]
                shift=versor((cos(angle),)+(0,)*(i-1)+(sin(angle),)+(0,)*(4+~i)) #versor((1/4+sqrt(5)/4,)+(0,)*(i-1)+(sqrt(5/8-sqrt(5)/8),)+(0,)*(4+~i)) #versor(((1+sqrt(5))/4,(sqrt(5)-1)/4,1/2,0)) #rotate 1/10th of the way around
                camera[0]=((shift*camera[0][0],camera[0][1]*shift**-1) if dims==4 else (camera[0][0],shift*camera[0][1]))
    oldToggles=toggles
    #tarmap(lambda i,o: drawShape(rad/(o[2] if dims==3 else sin(o[2])),o[:2],triangleWave(i/12),0),enumerate(tap(lambda e: toScreen(project(nodes[e])),edgeActions))) #0 is magenta for some reason (very suspicious) #4 (cyan) is forwards in my system, it seems
    drawLine((halfSize[0]-16,halfSize[1]),(halfSize[0]+16,halfSize[1]),0xffffff,4)
    drawLine((halfSize[0],halfSize[1]-16),(halfSize[0],halfSize[1]+16),0xffffff,4)
else: exit()
