import math, random
from functools import reduce
def conditionalReverse(reversal,toBe):
    return reversed(toBe) if reversal else toBe
def lap(func,*iterables): #Python 3 was a mistake
    return list(map(func,*iterables))
def dotProduct(a,b):
    return sum(map(float.__mul__,a,b))
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
        return tuple(math.sqrt(sum(c[1][i]**2*c[0] for c in colours)/sum(c[0] for c in colours)) for i in range(3))
    colours.insert(2,averageColours(*colours[:2]))
    for i in range(2):
        colours.insert(i+3,averageColours(colours[i],colours[2]))
    #light, dark, white, black, red, yellow, green
    global dims
    dims=3
    global size
    size=[1050]*dims
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
        screen.fill(black)
    global FPS
    FPS=60
import pygame
from pygame.locals import *
initialisePygame()
nodeNumber=8
rad=halfSize[0]/nodeNumber
nodes=[[[[i*rad]+halfSize[1:],[0]+[random.random()/2**8 for di in range(dims-1)]],(rad/4,)*dims, 1, angleColour(random.random()*math.pi)] for i in range(nodeNumber)]
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
def tangentCollision(i,j,r):
    differences=[di/r for di in map(float.__sub__,i[0][0],j[0][0])]
    magnitudes=(dotProduct(i[0][1],differences),dotProduct(j[0][1],differences))
    return [[(a-m)*di for di in differences] for m,a in zip(magnitudes,axialCollision(i[2],magnitudes[0],j[2],magnitudes[1]))]

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
                    discriminant=bee**2-a*(sum((ni-nj)**2 for ni,nj in zip(k[0][0],l[0][0]))-radius**2) #4*a*c part could instead be float.__mul__(*(sum((ni-nj)**2 for ni,nj in zip(k[0][m],nodes[j][0][m])) for m in range(2)))
                    if discriminant>0:
                        discriminant=math.sqrt(discriminant)
                        times=[(-bee+m*discriminant)/a for m in range(-1,3,2)] #all coefficients cancel out if you think about it
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
                for di,vi in enumerate(v):
                    nodes[i][0][1][di]+=vi
            pairsAlreadyCollided=[(i,j) for i,j in pairsAlreadyCollided if i not in bestCandidateIndices and j not in bestCandidateIndices]+[bestCandidateIndices]

def quaternionMultiply(a,b):
    return [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
            a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
            a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
            a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]] #this function not to be taken to before 1843

def quaternionConjugate(q): #usually conjugation is inverting the imaginary parts but as all quaternion enthusiasts know, inverting all four axes is ineffectual so inverting the first is ineffectually different from inverting the other three 
    return [-q[0]]+q[1:4] #(if you care about memory and not computing power, only the other three need to be stored with their polarities relative to it and the first's sign can be fixed and its magnitude computed (because it's a unit vector))

def rotateVector(v,q):
    return [(2*(q[2]**2+q[3]**2)-1)*v[0]+2*((q[0]*q[3]-q[1]*q[2])*v[1]-(q[0]*q[2]+q[1]*q[3])*v[2]), 
            (2*(q[1]**2+q[3]**2)-1)*v[1]+2*((q[0]*q[1]-q[2]*q[3])*v[2]-(q[1]*q[2]+q[0]*q[3])*v[0]),
            (2*(q[1]**2+q[2]**2)-1)*v[2]+2*((q[0]*q[2]-q[1]*q[3])*v[0]-(q[0]*q[1]+q[2]*q[3])*v[1])]

pixelAngle=math.tau/max(size)
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
        (i,j)=[math.atan2(i, z) for i in (x,y)]
    elif projectionMode==1: #azimuthal equidistant
        h=math.atan2((x**2+y**2),z**2)/math.hypot(x,y)
        return (x*h*minSize+halfSize[0],y*h*minSize+halfSize[1],math.hypot(*position))
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
    return (math.sin(direction)*magnitude*minSize+halfSize[0],math.cos(direction)*magnitude*minSize+halfSize[1],(math.hypot(*position) if projectionMode!=3 or radius==0 else radius))

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
toggleKeys=[pygame.K_z]
oldToggles=[False]*len(toggleKeys)
run=True
def normalise(vector):
    magnitude=math.sqrt(sum(map(abs,vector)))
    return (vector if magnitude==0 or magnitude==1 else [i/magnitude for i in vector])
def average(*vectors):
    l=len(vectors)
    return [sum(i)/l for i in zip(*vectors)]
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
    oldToggles=toggles
    if dims==3:
        if mouse.get_pressed()[0]:
            mouseChange=mouse.get_rel()
            mouseChange=(-mouseChange[1],mouseChange[0],0)
        else:
            mouse.get_rel() #otherwise it jumps
            mouseChange=(0,)*3
        cameraAngle[1]=[(di+acc*gain*angularVelocityConversionFactor)/(1+drag) for di,acc in zip(cameraAngle[1],normalise([keys[pygame.K_DOWN]-keys[pygame.K_UP],keys[pygame.K_LEFT]-keys[pygame.K_RIGHT],keys[pygame.K_q]-keys[pygame.K_e]]))]
        cameraAngle[0]=rotateByScreen(cameraAngle[0],[ci+mi for ci,mi in zip(cameraAngle[1],mouseChange)])
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
    for i,j,n in [(i,nodeScreenPositions[i],nodes[i]) for i in renderOrder]:
        sezi=2*((j[2] if projectionMode==3 else n[1][0]*minSize/j[2]) if perspectiveMode else n[1][0]) #different from size
        drawShape(sezi,j[:2],n[3],5)
else: exit()
