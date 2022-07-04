import math, random
from functools import reduce
def conditionalReverse(reversal,toBe):
    return reversed(toBe) if reversal else toBe
def lap(func,*iterables): #Python 3 was a mistake
    return list(map(func,*iterables))
def dotProduct(a,b):
    return sum(map(float.__mul__,a,b))
def sgn(n):
    return 1 if n>0 else -1 if n<0 else 0
def initialisePygame():
    global clock
    clock=pygame.time.Clock()
    pygame.init()
    global black
    black=(0,0,0)
    global colours
    colours=[(236,217,185),(174,137,104),(255,255,255),(0,0,0),(255,0,0),(255,255,0),(0,255,0)] #using lichess's square colours but do not tell lichess
    global angleColour
    def angleColour(angle):
        return tuple((math.cos(angle-math.tau*i/3)+1)*255/2 for i in range(3))
    global averageColours
    def averageColours(*colours):
        return tuple(math.sqrt(sum(c[i]**2 for c in colours)/len(colours)) for i in range(3)) #correct way, I think
    global weightedAverageColours
    def weightedAverageColours(*colours):
        return tuple(math.sqrt(sum(c[1][i]**2*abs(c[0]) for c in colours)/sum(c[0] for c in colours)) for i in range(3))
    colours.insert(2,averageColours(*colours[:2]))
    for i in range(2):
        colours.insert(i+3,averageColours(colours[i],colours[2]))
    #light, dark, white, black, red, yellow, green
    global dims
    dims=3
    global size
    size=[2560,1050]+[1050]*(dims-2)
    global minSize
    minSize=min(size[:2])
    global halfSize
    halfSize=[s/2 for s in size]
    global screen
    screen = pygame.display.set_mode(size[:2],pygame.RESIZABLE)
    global drawShape
    def drawShape(size,pos,colour,shape):
        if shape==0:
            pygame.draw.rect(screen,colour,pos+size)
        elif shape<5:
            pygame.draw.polygon(screen,colour,[[p+s/2*math.cos(((i+shape/2)/(2 if shape==4 else 4)+di/2)*math.pi) for di,(p,s) in enumerate(zip(pos,size))] for i in range(4 if shape==4 else 8)])
        else:
            pygame.draw.circle(screen,colour,pos,size/2)
    global drawLine
    def drawLine(initial,destination,colour):
        pygame.draw.line(screen,colour,initial,destination)

    global mouse
    mouse=pygame.mouse
    global doEvents
    def doEvents():
        global clickDone
        global run
        clickDone=False
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                run=False
            if event.type==pygame.MOUSEBUTTONUP:
                clickDone=1
            if event.type==pygame.WINDOWRESIZED:
                size[:2]=screen.get_rect().size
                minSize=min(size[:2])
                halfSize[:2]=[s/2 for s in size[:2]]
        pygame.display.flip()
        clock.tick(FPS)
        if not raytracingMode:
            screen.fill(black)
    global FPS
    FPS=60
import pygame
from pygame.locals import *
initialisePygame()
nodeNumber=64
rad=halfSize[0]/nodeNumber
nodes=[[[[i*rad]+halfSize[1:],[0]+[random.random()/2**8 for di in range(dims-1)]],(rad/4,)*dims, 1, angleColour(random.random()*2*math.pi)+(random.random(),)] for i in range(nodeNumber)]
stateTransitions=[[]]*len(nodes)
#each formatted [position,size,mass,colours]
def averageNode():
    return [sum(i[0][0][di] for i in nodes)/len(nodes)-si for di,si in enumerate(halfSize)]
cameraPosition=[averageNode(),[0.0]*3]
 #must be 0.0, not 0 (to be float for the lap(float.__add__))
cameraAngle=[[1/3,-1/6,math.sqrt(3)/2,-1/3],[0]*3] #these values make it point towards the diagram (there are probably ones that aren't askew but I like it (it has soul))
drag=0
gravitationalConstant=(size[0]/1000)*(64/len(nodes))**2
hookeStrength=1/size[0]
def physics():
    for i in nodes:
        if drag>0:
            absVel=max(1,math.hypot(*i[0][1])) #each dimension's deceleration from drag is its magnitude as a component of the unit vector of velocity times absolute velocity squared, is actual component times absolute velocity.
            i[0][1]=[di*(1-absVel*drag) for di in i[0][1]] #air resistance
    for i,(it,k) in enumerate(zip(stateTransitions[:-1],nodes[:-1])): #TypeError: 'zip' object is not subscriptable (I hate it so much)
        for j,(jt,l) in enumerate(zip(stateTransitions[i+1:],nodes[i+1:]),start=i+1):
            differences=lap(float.__sub__,l[0][0],k[0][0])
            #distSquare=sum(di**2 for di in differences)
            gravity=gravitationalConstant/max(2,sum(di**2 for di in differences)**1.5) #inverse-square law is 1/distance**2, for x axis is cos(angle of distance from axis)/(absolute distance)**2, the cos is x/(absolute), so is x/(abs)**3, however the sum outputs distance**2 so is exponentiated by 1.5 instead of 3
            gravity=(gravity*l[2],gravity*k[2])
            for ni,(ki,li,di) in enumerate(zip(k[0][1],l[0][1],differences)): #we are the knights who say ni
                nodes[i][0][1][ni]+=di*(hookeStrength*(j in it)+gravity[0])
                nodes[j][0][1][ni]-=di*(hookeStrength*(i in jt)+gravity[1])

def axialCollision(m0,v0,m1,v1):
    return (((m0-m1)*v0+2*m1*v1)/(m0+m1),
            ((m1-m0)*v1+2*m0*v0)/(m1+m0))
def tangentCollision(i,j,r,rayMode=False,lightSpeed=0):
    if rayMode:
        differences=[di/r for di in map(float.__sub__,i[0][0],j[0])]
        #magnitude=dotProduct(j[1],differences)
        #collision=dotProduct(i[0][1],differences)*2-magnitude #=((0-i[2])*magnitudes[1]+2*i[2]*magnitudes[0])/(0+i[2])
        #return [(collision-magnitude)*di for di in differences]
        collision=2*dotProduct((map(float.__sub__,i[0][1],j[1]) if lightSpeed else map(float.__neg__,j[1])),differences) #=dotProduct(i[0][1],differences)-dotProduct(j[1],differences)
        return (collision,[collision*di for di in differences])
    else:
        differences=[di/r for di in map(float.__sub__,i[0][0],j[0][0])]
        magnitudes=(dotProduct(i[0][1],differences),dotProduct(j[0][1],differences))
        return [[vi+(a-m)*di for vi,di in zip(v,differences)] for v,m,a in zip((i[0][1],j[0][1]),magnitudes,axialCollision(i[2],magnitudes[0],j[2],magnitudes[1]))]

def subframes():
    timeElapsed=0
    pairsAlreadyCollided=[]
    while timeElapsed<1:
        bestCandidateTime=1-timeElapsed
        bestCandidateIndices=(-1)*2
        for i,(it,k) in enumerate(zip(stateTransitions[:-1],nodes[:-1])): #TypeError: 'zip' object is not subscriptable (I hate it so much)
            for j,(jt,l) in enumerate(zip(stateTransitions[i+1:],nodes[i+1:]),start=i+1):
                if (i,j) not in pairsAlreadyCollided:
                    #(x+xv*t)**2+(y+yv*t)**2+(z+zv*t)**2=r**2
                    #x**2+2*x*xv*t+(xv*t)**2+y**2+2*y*yv*t+(yv*t)**2+z**2+2*z*zv*t+(zv*t)**2=r**2
                    #(xv**2+yv**2+zv**2)*t**2+2*(x*xv+y*yv+z*zv)*t+x**2+y**2+z**2-r**2=0
                    #t=(-2*(x*xv+y*yv+z*zv)±math.sqrt(4*(x*xv+y*yv+z*zv)**2-4*(xv**2+yv**2+zv**2)*(x**2+y**2+z**2-r**2)))/2*(xv**2+yv**2+zv**2)
                    #discriminant 4*(x*xv+y*yv+z*zv)**2-4*(xv**2+yv**2+zv**2)*(x**2+y**2+z**2-r**2)>=0
                    bee=sum((ni-nj)*(niv-njv) for (ni,niv,nj,njv) in zip(*k[0],*l[0])) #bzz (bee is half of b)
                    a=sum((ni-nj)**2 for ni,nj in zip(k[0][1],l[0][1]))
                    radius=k[1][0]+l[1][0]
                    discriminant=bee**2-a*(sum((ni-nj)**2 for ni,nj in zip(k[0][0],l[0][0]))-radius**2) #4*a*c part could instead be float.__mul__(*(sum((ni-nj)**2 for ni,nj in zip(k[0][m],nodes[j][0][m])) for m in range(2))) if a weren't to be reused
                    if discriminant>0:
                        discriminant=math.sqrt(discriminant)
                        times=[(-bee+p*discriminant)/a for p in range(-1,3,2)] #all coefficients cancel out if you think about it
                        candidate=times[times[0]<0]
                        if 0<candidate<bestCandidateTime:
                            bestCandidateTime=candidate
                            bestCandidateRadius=radius
                            bestCandidateNodes=(k,l)
                            bestCandidateIndices=(i,j)
        for i in nodes:
            i[0][0]=[s+v*bestCandidateTime for (s,v) in zip(*i[0])] #formerly lap(float.__add__,*i[0]) (back in my day before we had to bother with all of this sub-frame nonsense)
        timeElapsed+=bestCandidateTime

        if timeElapsed<1:
            #print(timeElapsed)
            for i,v in zip(bestCandidateIndices,tangentCollision(*bestCandidateNodes,bestCandidateRadius)):
                nodes[i][0][1]=v
            pairsAlreadyCollided=[(i,j) for i,j in pairsAlreadyCollided if i not in bestCandidateIndices and j not in bestCandidateIndices]+[bestCandidateIndices]

def quaternionMultiply(a,b):
    return [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
            a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
            a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
            a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]] #this function not to be taken to before 1843

def quaternionConjugate(q): #usually conjugation is inverting the imaginary parts but as all quaternion enthusiasts know, inverting all four axes is ineffectual so inverting the first is ineffectually different from inverting the other three
    return [-q[0]]+q[1:4] #(if you care about memory and not computing power, only the other three need to be stored with their polarities relative to the real component, which can have its sign fixed and its magnitude computed (because it's a unit vector))
#the reciprocal is the conjugate for unit vectors, otherwise divide by the magnitude squared

def rotateVector(v,q):
    return [(2*(q[2]**2+q[3]**2)-1)*v[0]+2*((q[0]*q[3]-q[1]*q[2])*v[1]-(q[0]*q[2]+q[1]*q[3])*v[2]), 
            (2*(q[1]**2+q[3]**2)-1)*v[1]+2*((q[0]*q[1]-q[2]*q[3])*v[2]-(q[1]*q[2]+q[0]*q[3])*v[0]),
            (2*(q[1]**2+q[2]**2)-1)*v[2]+2*((q[0]*q[2]-q[1]*q[3])*v[0]-(q[0]*q[1]+q[2]*q[3])*v[1])]

def eulerRotate(q,x,y,z): #rotates quaternion by Euler axes
    #equivalent to quaternionMultiply(quaternionMultiply(quaternionMultiply(q,(x[0],x[1],0,0)),(y[0],0,y[1],0)),(z[0],0,0,z[1])), or with
    #(x,y,z)=((math.cos(x),math.sin(x)),(math.cos(y),math.sin(y)),(math.cos(z),math.sin(z))) #equivalent to quaternionMultiply(quaternionMultiply(quaternionMultiply(q,(math.cos(x),math.sin(x),0,0)),(math.cos(y),0,math.sin(y),0)),(math.cos(z),0,0,math.sin(z)))
    return [((q[0]*x[0]-q[1]*x[1])*y[0]-(q[2]*x[0]+q[3]*x[1])*y[1])*z[0]-((q[3]*x[0]+q[2]*x[1])*y[0]+(q[1]*x[0]+q[0]*x[1])*y[1])*z[1],
            ((q[1]*x[0]+q[0]*x[1])*y[0]-(q[3]*x[0]+q[2]*x[1])*y[1])*z[0]+((q[2]*x[0]+q[3]*x[1])*y[0]+(q[0]*x[0]-q[1]*x[1])*y[1])*z[1],
            ((q[2]*x[0]+q[3]*x[1])*y[0]+(q[0]*x[0]-q[1]*x[1])*y[1])*z[0]-((q[1]*x[0]+q[0]*x[1])*y[0]-(q[3]*x[0]+q[2]*x[1])*y[1])*z[1],
            ((q[3]*x[0]+q[2]*x[1])*y[0]+(q[1]*x[0]+q[0]*x[1])*y[1])*z[0]+((q[0]*x[0]-q[1]*x[1])*y[0]-(q[2]*x[0]+q[3]*x[1])*y[1])*z[1]]

def slerp(a,b,t): #frankly delicious
    '''def arg(q):
        return math.acos(q[0]/math.sqrt(q[0]**2+q[1]**2+q[2]**2+q[3]**2))
    def ln(q):
        imaginaryMagnitude=math.sqrt(1-q[0]**2) #=q[1]**2+q[2]**2+q[3]**2
        imaginaryCoefficient=math.acos(q[0])/imaginaryMagnitude #the q[0] in the acos would be divided by magnitude if it weren't 1 (due to being a unit vector)
        return [0,imaginaryCoefficient*q[1],imaginaryCoefficient*q[2],imaginaryCoefficient*q[3]] #0 would be math.log(magnitude)
    def eToThe(q):
        eToTheReal=math.e**q[0]
        theSquares=math.sqrt(q[1]**2+q[2]**2+q[3]**2) #cannot be math.sqrt(1-q[0]**2) due to logarithms not being unit vectors
        imaginaryCoefficient=eToTheReal*(1 if theSquares==0 else math.sin(theSquares)/theSquares) #>sin(x)/x = lim x->0 (sin(x)/x) = 1 because... because it just does, okay
        return [eToTheReal*math.cos(theSquares),imaginaryCoefficient*q[1],imaginaryCoefficient*q[2],imaginaryCoefficient*q[3]]
    def toThe(q,p): #exponentiate quaternion by real power
        #return eToThe([p*n for n in ln(q)])
        imaginaryMagnitude=math.sqrt(1-q[0]**2)
        lnImaginaryCoefficient=math.acos(q[0])/imaginaryMagnitude
        #[0,imaginaryCoefficient*q[1],imaginaryCoefficient*q[2],imaginaryCoefficient*q[3]]
        #eToTheReal=1
        theSquares=p*lnImaginaryCoefficient*math.sqrt(imaginaryMagnitude)
        expImaginaryCoefficient=p*(1 if theSquares==0 else math.sin(theSquares)/theSquares)
        return [math.cos(theSquares),expImaginaryCoefficient*lnImaginaryCoefficient*q[1],expImaginaryCoefficient*lnImaginaryCoefficient*q[2],expImaginaryCoefficient*lnImaginaryCoefficient*q[3]]''' #elementary functions from which it's derived
    #equivalent to
    #return quaternionMultiply(a,toThe(quaternionMultiply(quaternionConjugate(a),b),t))
    #return quaternionMultiply(a,eToThe(t*n for n in ln(quaternionMultiply(quaternionConjugate(a),b))))
    """qzero=a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3]
    imaginaryMagnitude=1-qzero**2
    lnImaginaryCoefficient=math.acos(qzero)/imaginaryMagnitude
    '''return quaternionMultiply(a,eToThe([0,
                                        t*lnImaginaryCoefficient*(a[0]*b[1]-a[1]*b[0]-a[2]*b[3]+a[3]*b[2]),
                                        t*lnImaginaryCoefficient*(a[0]*b[2]+a[1]*b[3]-a[2]*b[0]-a[3]*b[1]),
                                        t*lnImaginaryCoefficient*(a[0]*b[3]-a[1]*b[2]+a[2]*b[1]-a[3]*b[0])]))'''
    #eToTheReal=1
    #theSquares=math.sqrt(q[1]**2+q[2]**2+q[3]**2) #=math.acos(qzero)
    #expImaginaryCoefficient=(1 if qzero==1 else expImaginaryCoefficient=math.sin(t*math.acos(qzero))/(t*math.acos(qzero))) #theSquares==math.acos(qzero)==0 means qzero==1
    #you could do expImaginaryCoefficient=(1 if qzero==1 else math.sqrt(1-qzero**2)/math.acos(qzero)) if there were no t coefficient
    #coeff=expImaginaryCoefficient*lnImaginaryCoefficient
    #coeff=(1 if qzero==1 else math.sin(t*math.acos(qzero))/(t*math.acos(qzero)))*math.acos(qzero)/imaginaryMagnitude
    #coeff=(1 if qzero==1 else math.sin(t*math.acos(qzero))/(t*math.acos(qzero)))*math.acos(qzero)/imaginaryMagnitude
    #coeff=((math.acos(qzero)/imaginaryMagnitude):=0 if qzero==1 else math.sin(t*math.acos(qzero))/(t*math.acos(qzero))*math.acos(qzero)/imaginaryMagnitude)
    coeff    =(1 if t==0 or qzero==1 else math.cos(t*math.acos(qzero))/(t*math.acos(qzero)))
    realCoeff=(0 if t==0 or qzero==1 else math.sin(t*math.acos(qzero)))
    '''return quaternionMultiply(a,[qzero,
                                 coeff*(a[0]*b[1]-a[1]*b[0]-a[2]*b[3]+a[3]*b[2]),
                                 coeff*(a[0]*b[2]+a[1]*b[3]-a[2]*b[0]-a[3]*b[1]),
                                 coeff*(a[0]*b[3]-a[1]*b[2]+a[2]*b[1]-a[3]*b[0])])'''""" #all bad (first attempt)
    qzero=a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3] #if negative, negate both this and b
    '''if qzero<0:
        b=lap(float.__neg__,b)''' #would be necessary (to prevent indirect paths when qzero sgn negative) but multiplying coeff by qzero's sgn fixes it also
    #imaginaryMagnitude=math.sqrt(1-qzero**2)
    theSquares=t*math.acos(abs(qzero)) #/imaginaryMagnitude and *imaginaryMagnitude cancel out
    #coeff*=t*(1 if theSquares==0 else math.sin(theSquares)/theSquares)/imaginaryMagnitude #do it later
    #coeff*=t*(1 if theSquares==0 else math.sin(theSquares)/(t*coeff))/imaginaryMagnitude
    coeff=(0 if theSquares==0 else math.sin(theSquares))/math.sqrt(1-qzero**2)*sgn(qzero) #could instead be divided by ((a[0]*b[1]-a[1]*b[0]-a[2]*b[3]+a[3]*b[2])**2+(a[0]*b[2]+a[1]*b[3]-a[2]*b[0]-a[3]*b[1])**2+(a[0]*b[3]-a[1]*b[2]+a[2]*b[1]-a[3]*b[0])**2)
    #theCos=math.cos(theSquares)
    '''return [a[0]*theCos-a[1]*coeff*(a[0]*b[1]-a[1]*b[0]-a[2]*b[3]+a[3]*b[2])-a[2]*coeff*(a[0]*b[2]+a[1]*b[3]-a[2]*b[0]-a[3]*b[1])-a[3]*coeff*(a[0]*b[3]-a[1]*b[2]+a[2]*b[1]-a[3]*b[0]),
            a[0]*coeff*(a[0]*b[1]-a[1]*b[0]-a[2]*b[3]+a[3]*b[2])+a[1]*theCos+a[2]*coeff*(a[0]*b[3]-a[1]*b[2]+a[2]*b[1]-a[3]*b[0])-a[3]*coeff*(a[0]*b[2]+a[1]*b[3]-a[2]*b[0]-a[3]*b[1]),
            a[0]*coeff*(a[0]*b[2]+a[1]*b[3]-a[2]*b[0]-a[3]*b[1])-a[1]*coeff*(a[0]*b[3]-a[1]*b[2]+a[2]*b[1]-a[3]*b[0])+a[2]*theCos+a[3]*coeff*(a[0]*b[1]-a[1]*b[0]-a[2]*b[3]+a[3]*b[2]),
            a[0]*coeff*(a[0]*b[3]-a[1]*b[2]+a[2]*b[1]-a[3]*b[0])+a[1]*coeff*(a[0]*b[2]+a[1]*b[3]-a[2]*b[0]-a[3]*b[1])-a[2]*coeff*(a[0]*b[1]-a[1]*b[0]-a[2]*b[3]+a[3]*b[2])+a[3]*theCos]''' #simplifies to
    '''return [a[0]*theCos+coeff*(b[0]*(a[1]**2+a[2]**2+a[3]**2)-a[0]*(a[1]*b[1]+a[2]*b[2]+a[3]*b[3])),
            a[1]*theCos+coeff*(b[1]*(a[0]**2+a[2]**2+a[3]**2)-a[1]*(a[0]*b[0]+a[2]*b[2]+a[3]*b[3])),
            a[2]*theCos+coeff*(b[2]*(a[0]**2+a[1]**2+a[3]**2)-a[2]*(a[0]*b[0]+a[1]*b[1]+a[3]*b[3])),
            a[3]*theCos+coeff*(b[3]*(a[0]**2+a[1]**2+a[2]**2)-a[3]*(a[0]*b[0]+a[1]*b[1]+a[2]*b[2]))]''' #then, because a[0]**2+a[1]**2+a[2]**2+a[3]**2=1 and qzero=a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3],
    '''return [a[0]*theCos+coeff*(b[0]*(1-a[0]**2)-a[0]*(qzero-a[0]*b[0])),
            a[1]*theCos+coeff*(b[1]*(1-a[1]**2)-a[1]*(qzero-a[1]*b[1])),
            a[2]*theCos+coeff*(b[2]*(1-a[2]**2)-a[2]*(qzero-a[2]*b[2])),
            a[3]*theCos+coeff*(b[3]*(1-a[3]**2)-a[3]*(qzero-a[3]*b[3]))]''' #I like this part
    '''return [a[0]*theCos+coeff*(b[0]-a[0]*qzero),
            a[1]*theCos+coeff*(b[1]-a[1]*qzero),
            a[2]*theCos+coeff*(b[2]-a[2]*qzero),
            a[3]*theCos+coeff*(b[3]-a[3]*qzero)]'''
    theCos=math.cos(theSquares)-coeff*qzero #qzero's negativeness will cancel out b's if it (the dot product) is negative, thus no negation necessary
    return [a[0]*theCos+coeff*b[0],
            a[1]*theCos+coeff*b[1],
            a[2]*theCos+coeff*b[2],
            a[3]*theCos+coeff*b[3]]
#print([math.hypot(*slerp([1,0,0,0],[1/3,-1/6,math.sqrt(3)/2,-1/3],n/100)) for n in range(100)]) #test that it yields intermediate unit vectors
#print([slerp([1,0,0,0],[1/3,-1/6,math.sqrt(3)/2,-1/3],n/4) for n in range(5)])
#print([slerp([1,0,0,0],[-1/3,-1/6,math.sqrt(3)/2,-1/3],n/4) for n in range(5)]) #test that it takes optimal route

pixelAngle=1/minSize #math.tau/max(size)
def rotateByScreen(angle,screenRotation):
    magnitude=math.hypot(*screenRotation)
    if magnitude==0:
        return angle
    else:
        magpi=magnitude*pixelAngle
        simagomag=math.sin(magpi)/magnitude #come and get your simagomags delivered fresh
        return quaternionMultiply([math.cos(magpi)]+[i*simagomag for i in screenRotation], angle)

projectionMode=2
def projectRelativeToScreen(position,radius=rad): #input differences from player position after rotation applied #radius only necessary for stereographic
    (x,y,z)=position
    if projectionMode==0: #weird
        return [math.atan2(i, z) for i in (x,y)]
    elif projectionMode==1: #azimuthal equidistant
        h=math.atan2((x**2+y**2),z**2)/math.hypot(x,y)
        return (x*h/pixelAngle+halfSize[0],y*h/pixelAngle+halfSize[1],math.hypot(*position))
    else:
        magnitude=math.atan2(math.hypot(x,y),z) #other azimuthals
        if projectionMode==2: #Lambert equi-area (not to be taken to before 1772)
            magnitude=2*abs(math.sin(magnitude/2)) #formerly math.sqrt(math.sin(magnitude)**2+(math.cos(magnitude)-1)**2)
        elif projectionMode==3: #stereographic (trippy)
            if radius==0:
                magnitude=1/math.tan(magnitude/2) #equivalent to math.sin(magnitude)/(1-math.cos(magnitude))
            else: #doesn't only find centre but uses fact that circles on the unit sphere of the camera's eye are projected to circles on the plane
                h=math.hypot(x,y)
                hh=math.hypot(*position)
                offset=math.asin(radius/hh)
                s0=1/math.tan((magnitude-offset)/2) #>mfw no math.cot
                s1=1/math.tan((magnitude+offset)/2)
                radius=s1-s0
                magnitude=(s0+s1)/2
        direction=math.atan2(x,y)
        return (math.sin(direction)*magnitude/pixelAngle+halfSize[0],math.cos(direction)*magnitude/pixelAngle+halfSize[1],(math.hypot(*position) if projectionMode!=3 or radius==0 else radius))

def antiProject(x,y,q): #project spherical coordinates to vector (its own function due to rotateVector beiing greatly optimised by unit vector inputs)
    #antiProject(x,y,q,cameraAngle[0]) is equivalent to rotateVector((0,0,1),rotateByScreen(cameraAngle[0],(x,y,0)))
    magnitude=math.hypot(x,y)
    magpi=magnitude*pixelAngle
    comag=math.cos(magpi)
    simagomag=math.sin(magpi)/magnitude
    [i*simagomag for i in (x,y,0)],cameraAngle[0]
    #rotateByScreen yields
    #return [comag*b[0]-simagomag*(x*b[1]+y*b[2]),
            #comag*b[1]+simagomag*(x*b[0]+y*b[3]),
            #comag*b[2]-simagomag*(x*b[3]-y*b[0]),
            #comag*b[3]+simagomag*(x*b[2]-y*b[1])]
    #rotateVector((0,0,1),q) yields
    #return [-2*(q[0]*q[2]+q[1]*q[3]),
             #2*(q[0]*q[1]-q[2]*q[3]),
             #2*(q[1]**2+q[2]**2)-1]
    return [-2*((comag*q[0]-simagomag*(x*q[1]+y*q[2]))*(comag*q[2]-simagomag*(x*q[3]-y*q[0]))+(comag*q[1]+simagomag*(x*q[0]+y*q[3]))*(comag*q[3]+simagomag*(x*q[2]-y*q[1]))),
             2*((comag*q[0]-simagomag*(x*q[1]+y*q[2]))*(comag*q[1]+simagomag*(x*q[0]+y*q[3]))-(comag*q[2]-simagomag*(x*q[3]-y*q[0]))*(comag*q[3]+simagomag*(x*q[2]-y*q[1]))),
             2*((comag*q[1]+simagomag*(x*q[0]+y*q[3]))**2+(comag*q[2]-simagomag*(x*q[3]-y*q[0]))**2)-1]

def rayAngle(x,y): #converts screen position to ray velocity vector
    if projectionMode==0:
        raise RuntimeError("the weird projection isn't supported for raytracing") #too weird
    else:
        x-=halfSize[0]
        y-=halfSize[1]
        magnitude=pixelAngle*math.hypot(x,y)
        #print(magnitude/math.sqrt(2))
        bagnitude=(1 if projectionMode==1 else math.asin(magnitude/math.sqrt(2))/magnitude if projectionMode==2 else 2*math.atan(1/magnitude)/magnitude) #if projectionMode==3 else)
        #direction=math.atan2(y,x)
        #(i,j)=(cos(direction)*bagnitude,sin(direction)*bagnitude) #bagnitude now divided by magnitude (instead of both i and j)
        return antiProject(x*bagnitude,y*bagnitude,cameraAngle[0]) #rotateVector((0,0,1),rotateByScreen(cameraAngle[0],(x*bagnitude,y*bagnitude,0)))

theThing=4*math.sqrt(3) #you know, the thing
def raytrace(hexLattice,exposure,upscale,fieldOfView,maxReflections=8,antialiasing=0,timeAntialiasing=0,lightSpeed=0,gravityMode=0):
    #hexLattice is for when rendering in low resolution in real time (decreases average screen pixel's distance from the closest ray), antialiasing only probabilistic for now, timeAntialiasing for motion blur, lightSpeed for simulating it being finite (0 for infinite), gravityMode imparts gravity from spheres upon rays at time intervals determined by its value (if nonzero), and makes reflections instead describe the number of steps
    pixelRadius=minSize*pixelAngle/min(fieldOfView)*upscale
    normalise=True #in finite-light-speed mode, if light reflects off a moving object, its velocity's magnitude will be renormalised
    jInc=math.sqrt(3)/2 if hexLattice else 1 #meaning j increment but perhaps also j incorporated
    for screenK,screenerinoJ in enumerate(range(int(fieldOfView[1]/pixelAngle/jInc/upscale))):
        screenJ=screenerinoJ*jInc*upscale #range doesn't work with float inputs
        for screenerinoI in range(int((fieldOfView[0]/pixelAngle+jInc*hexLattice*screenK%2)/upscale)):
            screenI=screenerinoI*upscale
            antialiasColour=[0,0,0]
            for antialiasPasses in range(antialiasing if antialiasing else 1):
                timeElapsed=0
                (i,j,t)=[random.random()-0.5 for _ in range(2+timeAntialiasing)]+[0]*(not timeAntialiasing) if antialiasing else (0,)*3
                rayPosition=[cameraPosition[0],rayAngle(screenI+i,screenJ+j)]
                if lightSpeed!=0:
                    rayPosition[1]=[n*lightSpeed for n in rayPosition[1]]
                reflectivenesses=[]
                reflectionColours=[]
                reflections=0
                bestCandidateIndice=-1
                while (reflections==0 or bestCandidateIndice!=-1 or gravityMode) and reflections<maxReflections:
                    if gravityMode:
                        gravityAcceleration=[0]*3
                    bestCandidateTime=(gravityMode-timeElapsed%gravityMode if gravityMode else 0)
                    bestCandidateIndice=-1
                    for m,n in enumerate(nodes):
                        spherePosition=[([s-timeElapsed*v for s,v in n[0]] if lightSpeed else n[0][0]),(n[0][1] if lightSpeed else [0]*3)] #subtract p (velocity) because rays are travelling back in time
                        differences=[s-r for s,r in zip(spherePosition[0],rayPosition[0])] #lap(float.__sub__,spherePosition[0],rayPosition[0]) causes TypeError: unsupported operand type(s) for *: 'NotImplementedType' and 'float'
                        bee=sum(di*(niv+njv) for (di,niv,njv) in zip(differences,spherePosition[1],rayPosition[1])) #always add sphere velocity to ray for relative instead of subtracting
                        a=sum((ni+nj)**2 for ni,nj in zip(spherePosition[1],rayPosition[1]))
                        radius=n[1][0]
                        discriminant=bee**2-a*(sum(di**2 for di in differences)-radius**2)
                        #print("disc",discriminant)
                        if discriminant>0:
                            discriminant=math.sqrt(discriminant)
                            times=[(-bee+p*discriminant)/a for p in range(-1,3,2)]
                            candidate=times[times[0]<0]
                            if 0<candidate and not 0!=bestCandidateTime>=candidate: #0<candidate and (bestCandidateTime==0 or candidate<bestCandidateTime):
                                bestCandidateTime=candidate
                                bestCandidateNode=n
                                bestCandidateIndice=m
                        if gravityMode:
                            gravity=gravityMode*gravitationalConstant/max(2,sum(di**2 for di in differences)**1.5)*n[2]
                            gravityAcceleration=[gi+gravity*di for gi,di in zip(gravityAcceleration,differences)]
                    if gravityMode:
                        rayPosition[1]=[v+bestCandidateTime*ga for v,ga in zip(rayPosition[1],gravityAcceleration)]
                    rayPosition[0]=[s+bestCandidateTime*v for s,v in zip(*rayPosition)]
                    if lightSpeed:
                        timeElapsed+=bestCandidateTime
                    if bestCandidateIndice!=-1:
                        #print("wibble")
                        (mag,newVelocity)=tangentCollision(bestCandidateNode,rayPosition,bestCandidateNode[1][1],True)
                        if normalise:
                            #speed=math.hypot(*newVelocity)
                            newVelocity=[v*lightSpeed/mag for v in newVelocity]
                            #collisionMagnitude=math.hypot(*(n-o for n,o in zip(newVelocity,rayPosition[1])))
                            rayPosition[1]=newVelocity
                            reflectivenesses.append(1-(1-bestCandidateNode[3][3]*mag/(lightSpeed if lightSpeed else 1)) if True else bestCandidateNode[3][3])
                            reflectionColours.append(bestCandidateNode[3][:3])
                    reflections+=1
                rayColour=[0,0,0]
                for r,c in zip(reflectivenesses[::-1],reflectionColours[::-1]):
                    rayColour=weightedAverageColours((1-r,rayColour),(r,c))
            antialiasColour=(weightedAverageColours((antialiasPasses/(antialiasPasses+1),antialiasColour),(1/(antialiasPasses+1),rayColour)) if antialiasing else rayColour)
            antialiasColour=[min(round(i),255) for i in antialiasColour]
            '''if antialiasColour!=[0]*3:
                print("radius",pixelRadius,"position",(screenI/(upscale)/(fieldOfView[1]/pixelAngle/jInc/upscale)*size[0],hexLattice/theThing-screenJ/(jInc*upscale)/((fieldOfView[0]/pixelAngle+jInc*hexLattice*screenK%2)/upscale)*size[1]),"colour",antialiasColour)
                print("(",screenI,"*",size[0],"/",fieldOfView[0],",",hexLattice,"/",theThing,"-",screenJ,"*",size[1],"/",fieldOfView[1],")")'''
            drawShape(pixelRadius,((screenI+1/2)/upscale/(fieldOfView[0]/pixelAngle/jInc/upscale)*size[0],hexLattice/theThing+(screenJ+1/2)/(jInc*upscale)/((fieldOfView[1]/pixelAngle+jInc*hexLattice*screenK%2)/upscale)*size[1]),antialiasColour,5)


def findNodeScreenPositions():
    output=[n[0][0] for n in nodes]
    if dims==3:
        output=[rotateVector(lap(float.__sub__,n,cameraPosition[0]),cameraAngle[0]) for n in output]
        if perspectiveMode:
            output=lap(projectRelativeToScreen,output)
    return output

gain=1
angularVelocityConversionFactor=math.tau/FPS
perspectiveMode=True
physicsMode=True
raytracingMode=False
toggleKeys=[pygame.K_z,pygame.K_t]
oldToggles=[False]*len(toggleKeys)
run=True
def normalise(vector):
    magnitude=math.sqrt(sum(map(abs,vector)))
    return (vector if magnitude==0 or magnitude==1 else [i/magnitude for i in vector])
def average(*vectors):
    l=len(vectors)
    return [sum(i)/l for i in zip(*vectors)]
frames=0
while run:
    keys=pygame.key.get_pressed()
    doEvents()
    if perspectiveMode:
        cameraPosition=[lap(float.__add__,*cameraPosition),([0.0]*3 if keys[pygame.K_LSHIFT] else lap(float.__add__,cameraPosition[1],rotateVector(normalise([keys[pygame.K_d]-keys[pygame.K_a],keys[pygame.K_f]-keys[pygame.K_r],keys[pygame.K_w]-keys[pygame.K_s]]),quaternionConjugate(cameraAngle[0]))))]
    else:
        cameraPosition[0]=averageNode()
    toggles=[keys[k] for k in toggleKeys]
    for i,(k,o) in enumerate(zip(toggles,oldToggles)):
        if k==0!=o:
            if i==0: #space (between turn to move, winningness and DTM)
                physicsMode^=True #(because it can be in real time without O(n*(n-1)/2) gravity simulation)
            elif i==1:
                raytracingMode^=True
    oldToggles=toggles
    if dims==3:
        if mouse.get_pressed()[0]:
            mouseChange=mouse.get_rel()
            mouseChange=mouseChange+(0,) if raytracingMode else (-mouseChange[1],mouseChange[0],0)
        else:
            mouse.get_rel() #otherwise it jumps
            mouseChange=(0,)*3
        cameraAngle[1]=([0.0]*3 if keys[pygame.K_LSHIFT] and drag==0 else [(di+acc*gain*angularVelocityConversionFactor)/(1+drag) for di,acc in zip(cameraAngle[1],normalise([keys[pygame.K_DOWN]-keys[pygame.K_UP],keys[pygame.K_LEFT]-keys[pygame.K_RIGHT],keys[pygame.K_q]-keys[pygame.K_e]]))])
        cameraAngle[0]=rotateByScreen(cameraAngle[0],[ci+mi for ci,mi in zip(cameraAngle[1],mouseChange)]) #slerp([1,0,0,0],[1/3,-1/6,math.sqrt(3)/2,-1/3],frames/100) #(for testing)
    nodeScreenPositions=findNodeScreenPositions()
    renderOrder=conditionalReverse(perspectiveMode,[j for _, j in sorted((p[2],i) for i,p in enumerate(nodeScreenPositions))]) #perhaps replace with zip(*sorted((p[2],i) for i,p in enumerate(nodeScreenPositions)))[1] (not sure)
    if physicsMode:
        physics()
        subframes()
    for sc,k,n in zip(nodeScreenPositions,stateTransitions,nodes):
        for l in k:
            drawLine(sc[:2],(average(sc[:2],nodeScreenPositions[l][:2]) if directedMode else nodeScreenPositions[l][:2]),n[3][colourMode])
    if clickDone:
        mousePos=mouse.get_pos()
    if raytracingMode:
        raytrace(False,1/4,100,(size[0]/size[1],1),maxReflections=8,antialiasing=0,timeAntialiasing=0,lightSpeed=0,gravityMode=0)
    else:
        for i,j,n in [(i,nodeScreenPositions[i],nodes[i]) for i in renderOrder]:
            sezi=2*((j[2] if projectionMode==3 else n[1][0]/pixelAngle/j[2]) if perspectiveMode else n[1][0]) #different from size
            drawShape(sezi,j[:2],n[3],5)
    frames+=1
else: exit()
