import math, random
from functools import reduce
def conditionalReverse(reversal,toBe):
    return reversed(toBe) if reversal else toBe
def lap(func,*iterables): #Python 3 was a mistake
    return list(map(func,*iterables))
def dotProduct(a,b):
    return sum(map(float.__mul__,a,b))
def crossProduct(a,b):
    return (a[1]*b[2]-a[2]*b[1],
            a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0]) if dims==3 else a[0]*b[1]-a[1]*b[0] #very suspicious indeed
def drossProduct(a,b):
    return sum(a)*sum(b)-dotProduct(a,b)
def sgn(n):
    return 1 if n>0 else -1 if n<0 else 0

import pygame
from pygame.locals import *
clock=pygame.time.Clock()
pygame.init()
black=(0,0,0)
colours=[(236,217,185),(174,137,104),(255,255,255),(0,0,0),(255,0,0),(255,255,0),(0,255,0)] #using lichess's square colours but do not tell lichess
def angleColour(angle):
    return tuple((math.cos(angle-math.tau*i/3)+1)*255/2 for i in range(3))
def averageColours(*colours):
    return tuple(math.sqrt(sum(c[i]**2 for c in colours)/len(colours)) for i in range(3)) #correct way, I think
def weightedAverageColours(*colours):
    return tuple(math.sqrt(sum(c[1][i]**2*abs(c[0]) for c in colours)/sum(c[0] for c in colours)) for i in range(3))
colours.insert(2,averageColours(*colours[:2]))
for i in range(2):
    colours.insert(i+3,averageColours(colours[i],colours[2]))
#light, dark, white, black, red, yellow, green
gameMode=True
dims=2#+(not gameMode) #it will be a 2D game
playerNode=True
playerNodeOffset=((0.0,-40.0,-80.0) if dims==3 else (0.0,0.0))
size=[2560,1050]+[1050]*(dims-2)
minSize=min(size[:2])
halfSize=[s/2 for s in size]
screen=pygame.display.set_mode(size[:2],pygame.RESIZABLE)
def drawShape(size,pos,colour,shape):
    if shape==0:
        pygame.draw.rect(screen,colour,pos+size)
    elif shape<5:
        pygame.draw.polygon(screen,colour,[[p+s/2*math.cos(((i+shape/2)/(2 if shape==4 else 3 if shape==3 else  4)+di/2)*math.pi) for di,(p,s) in enumerate(zip(pos,size))] for i in range(4 if shape==4 else 6 if shape==3 else 8)])
    else:
        pygame.draw.circle(screen,colour,pos,size/2)
def drawLine(initial,destination,colour,width=1):
    if width!=1:
        #trigonometry
        #angle=math.atan2(*map(float.__sub__,destination,initial))%(math.pi/2)
        #angle=min(angle,math.pi/2-angle)
        #width=int(width/math.cos(angle))
        #optimised by 1/cos(atan(x/y)=sqrt((x/y)**2+1)
        #displacement=[(d-i)**2 for d,i in zip(destination,initial)]
        #displacement.sort()
        #if displacement[1]!=0.0:
            #width=int(width*(displacement[0]/displacement[1]+1)**0.5)
        #more optimised
        (x,y)=[(d-i)**2 for d,i in zip(destination,initial)]
        if x!=0 or y!=0:
            pygame.draw.line(screen,colour,initial,destination,int(width*((x/y if x<y else y/x)+1)**0.5)) #thank you WolframAlpha
    else:
        pygame.draw.line(screen,colour,initial,destination)

mouse=pygame.mouse
def doEvents():
    global run
    global clickDone
    clickDone=False
    global framesMouseDown
    global shortClickDone
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
            size[:2]=screen.get_rect().size
            minSize=min(size[:2])
            halfSize[:2]=[s/2 for s in size[:2]]
    pygame.display.flip()
    clock.tick(FPS)
    if not raytracingMode:
        screen.fill(black)
    size[:2]=screen.get_size()
FPS=60
nodeNumber=64
rad=halfSize[0]/nodeNumber
orthographicZoom=nodeNumber/8
spatialGain=1/(2*orthographicZoom) #would be *math.sqrt(2)**gameMode because in gameMode, the motion will be diagonal and normalised, but that makes shaking on the small circles too great

nodes=([[[[0.0]*dims]*2,(rad/4**gameMode,)*dims,1,(255,)*3,0]] if playerNode else [])
def averageNode():
    return [sum(i[0][0][di] for i in nodes)/len(nodes)-si for di,si in enumerate(halfSize)]
cameraPosition=[averageNode(),[0.0]*3]
#each formatted [position,size,mass,colour,fixedness] #fixed balls are fixed in-place but still create gravity
if gameMode:
    nodes.append([[[0.0,100.0]+[0.0]*(dims==3),[0.0]*dims],[50.0]*dims,(50.0/rad)**dims,(127,)*3,1])
    nodes+=[[[[p+100*math.cos((a*2/6-di/2)*math.pi) for di,p in enumerate(nodes[1][0][0])],[0.0]*dims],[10.0]*dims,(10.0/rad)**dims,(127,)*3,1] for a in range(6)]
else:
    nodes+=[[[[i*rad*4]+halfSize[1:],[0]+[random.random()/2**8 for di in range(dims-1)]],(rad,)*dims,1,angleColour(random.random()*2*math.pi)+(random.random(),),0] for i in range(nodeNumber)]
    nodes+=[[[[po+rad*4*((-1)**p)*(i==di) for i,po in enumerate(cameraPosition[0])],cameraPosition[1]],(rad,)*dims,0,angleColour(math.pi*(di/3+p))+(1/2,),1] for di in range(dims) for p in range(2)]
stateTransitions=[[]]*len(nodes)
cameraAngle=[((1.0,0.0,0.0,0.0) if True else [1/3,-1/6,math.sqrt(3)/2,-1/3]),[0.0]*3] #these values make it point towards the diagram (there are probably ones that aren't askew but I like it (it has soul))
#(1,0,0,0) points in negative y with negative x to the right and negative z to the up
drag=1/2**6*gameMode
gravitationalConstant=(size[0]/(100*(50**gameMode)))*(64/len(nodes))**2
hookeStrength=0 #1/size[0]
def normalise(vector,allowSmaller=False,resultantMagnitude=1):
    magnitude=math.hypot(*vector) #math.sqrt(sum(map(abs,vector)))
    return (vector if magnitude==0 or (magnitude<=resultantMagnitude if allowSmaller else magnitude==resultantMagnitude) else [i*resultantMagnitude/magnitude for i in vector])
def physics():
    for i in nodes:
        if drag>0:
            absVel=max(1,math.hypot(*i[0][1])) #each dimension's deceleration from drag is its magnitude as a component of the unit vector of velocity times absolute velocity squared, is actual component times absolute velocity.
            i[0][1]=[di*(1-absVel*drag) for di in i[0][1]] #air resistance
    if gameMode:
        playerInitial=list(nodes[0][0][1])
    for j,(jt,l) in enumerate(zip(stateTransitions[1:],nodes[1:]),start=1):  #TypeError: 'zip' object is not subscriptable (I hate it so much)
        for i,(it,k) in enumerate(zip(stateTransitions[:j],nodes[:j])):
            if not (i==0 and l[4]==2 or l[4]==3):
                differences=lap(float.__sub__,l[0][0],k[0][0])
                distSquare=sum(di**2 for di in differences)
                radSquare=(l[1][0]+k[1][0])**2 #"the square is rad, man"
                gravity=gravitationalConstant*( min(1/2,1/distSquare**1.5) #inverse-square law is 1/distance**2, for x axis is cos(angle of distance from axis)/(absolute distance)**2, the cos is x/(absolute), so is x/(abs)**3, however the sum outputs distance**2 so is exponentiated by 1.5 instead of 3
                                               if distSquare>radSquare else
                                                math.sqrt(distSquare/radSquare)/radSquare**1.5) #not quite correct because the shell theorem only works when the 
                gravity=(gravity*l[2],gravity*k[2])
                for ni,(ki,li,di) in enumerate(zip(k[0][1],l[0][1],differences)): #we are the knights who say ni
                    if nodes[i][4]!=1:
                        nodes[i][0][1][ni]+=di*(hookeStrength*(j in it)+gravity[0] if hookeStrength else gravity[0])
                    if nodes[j][4]!=1:
                        nodes[j][0][1][ni]-=di*(hookeStrength*(i in jt)+gravity[1] if hookeStrength else gravity[1])
    if gameMode:
        playerAcceleration=normalise(lap(float.__sub__,nodes[0][0][1],playerInitial))
        if playerAcceleration!=[0.0]*dims:
            global cameraAngle
            cameraAngle[0]=[playerAcceleration[1],playerAcceleration[0]]

def axialCollision(m0,v0,m1,v1):
    return (((m0-m1)*v0+2*m1*v1)/(m0+m1),
            ((m1-m0)*v1+2*m0*v0)/(m1+m0))
def tangentCollision(i,j,r,rayMode=False,lightSpeed=0,returnRayCollisionMagnitude=True,makeRayTangent=False):
    if rayMode:
        differences=[di/r for di in map(float.__sub__,i[0][0],j[0])]
        #magnitude=dotProduct(j[1],differences)
        #collision=dotProduct(i[0][1],differences)*2-magnitude #=((0-i[2])*magnitudes[1]+2*i[2]*magnitudes[0])/(0+i[2])
        #return [(collision-magnitude)*di for di in differences]
        collision=(1.0625 if makeRayTangent else 2)*dotProduct((map(float.__sub__,i[0][1],j[1]) if lightSpeed else map(float.__neg__,j[1])),differences) #=dotProduct(i[0][1],differences)-dotProduct(j[1],differences)
        return (collision,[ji+collision*di for ji,di in zip(j[1],differences)]) if returnRayCollisionMagnitude else [ji+collision*di for ji,di in zip(j[1],differences)]
    else:
        differences=[di/r for di in map(float.__sub__,i[0][0],j[0][0])]
        magnitudes=(dotProduct(i[0][1],differences),dotProduct(j[0][1],differences))
        #return [[vi+(a-m)*di for vi,di in zip(v,differences)] for v,m,a in zip((i[0][1],j[0][1]),magnitudes,axialCollision(i[2],magnitudes[0],j[2],magnitudes[1]))]
        iChange=((i[2]-j[2])*magnitudes[0]+2*j[2]*magnitudes[1])/(i[2]+j[2])-magnitudes[0]
        jChange=((j[2]-i[2])*magnitudes[1]+2*i[2]*magnitudes[0])/(j[2]+i[2])-magnitudes[1]
        return ([vi+iChange*di for vi,di in zip(i[0][1],differences)],
                [vi+jChange*di for vi,di in zip(j[0][1],differences)])

def subframes():
    global playerLanded
    global armExtended
    global tangentDivergence
    def deleteArm(armIndice):
        del nodes[armIndice]
        del stateTransitions[armIndice]
        global itsOver
        itsOver=True #only darkness now
        global armExtended
        armExtended=False
    playerLanded=False
    timeElapsed=0
    pairsAlreadyCollided=[]
    nodeFramesProceeded=[0.0]*len(nodes)
    while timeElapsed<1:
        bestCandidateTime=1-timeElapsed
        bestCandidateIndices=(-1,)*2
        for j,(jt,l,u) in enumerate(zip(stateTransitions[1:],nodes[1:],nodeFramesProceeded[1:]),start=1):
            for i,(it,k,t) in enumerate(zip(stateTransitions[:j],nodes[:j],nodeFramesProceeded[:j])):
                if (i,j) not in pairsAlreadyCollided or l[4]==3:
                    '''#(x+xv*t)**2+(y+yv*t)**2+(z+zv*t)**2=r**2
                    #x**2+2*x*xv*t+(xv*t)**2+y**2+2*y*yv*t+(yv*t)**2+z**2+2*z*zv*t+(zv*t)**2=r**2
                    #(xv**2+yv**2+zv**2)*t**2+2*(x*xv+y*yv+z*zv)*t+x**2+y**2+z**2-r**2=0
                    #t=(-2*(x*xv+y*yv+z*zv)±math.sqrt(4*(x*xv+y*yv+z*zv)**2-4*(xv**2+yv**2+zv**2)*(x**2+y**2+z**2-r**2)))/2*(xv**2+yv**2+zv**2)
                    #discriminant 4*(x*xv+y*yv+z*zv)**2-4*(xv**2+yv**2+zv**2)*(x**2+y**2+z**2-r**2)>=0
                    #t=(-(x*xv+y*yv+z*zv)±math.sqrt((x*xv+y*yv+z*zv)**2-(xv**2+yv**2+zv**2)*(x**2+y**2+z**2-r**2)))/(xv**2+yv**2+zv**2)
                    bee=sum((ni-nj)*(niv-njv) for (ni,niv,nj,njv) in zip(*k[0],*l[0])) #bzz (bee is half of b)
                    a=sum((ni-nj)**2 for ni,nj in zip(k[0][1],l[0][1]))
                    radius=k[1][0]+l[1][0]
                    discriminant=bee**2-a*(sum((ni-nj)**2 for ni,nj in zip(k[0][0],l[0][0]))-radius**2) #4*a*c part could instead be float.__mul__(*(sum((ni-nj)**2 for ni,nj in zip(k[0][m],nodes[j][0][m])) for m in range(2))) if a weren't to be reused''' #instead, store asynchronously
                    #((x1-x0)+(xv1*(t-t1)-xv0*(t-t0)))**2+((y1-y0)+(yv1*(t-t1)-yv0*(t-t0)))**2+((z1-z0)+(zv1*(t-t1)-zv0*(t-t0)))**2=r**2
                    #(x1-x0)**2+2*(x1-x0)*(xv1*(t-t1)-xv0*(t-t0))+(xv1*(t-t1)-xv0*(t-t0))**2+(y1-y0)**2+2*(y1-y0)*(yv1*(t-t1)-yv0*(t-t0))+(yv1*(t-t1)-yv0*(t-t0))**2+(z1-z0)**2+2*(z1-z0)*(zv1*(t-t1)-zv0*(t-t0))+(zv1*(t-t1)-zv0*(t-t0))**2=r**2
                    #(x1-x0)**2+(y1-y0)**2+(z1-z0)**2+2*((x1-x0)*(xv1*(t-t1)-xv0*(t-t0))+(y1-y0)*(yv1*(t-t1)-yv0*(t-t0))+(z1-z0)*(zv1*(t-t1)-zv0*(t-t0)))+(xv1*(t-t1)-xv0*(t-t0))**2+(yv1*(t-t1)-yv0*(t-t0))**2+(zv1*(t-t1)-zv0*(t-t0))**2=r**2
                    #(x1-x0)**2+(y1-y0)**2+(z1-z0)**2+2*((x1-x0)*xv1*(t-t1)-(x1-x0)*xv0*(t-t0)+(y1-y0)*yv1*(t-t1)-(y1-y0)*yv0*(t-t0)+(z1-z0)*zv1*(t-t1)-(z1-z0)*zv0*(t-t0))+(xv1*(t-t1)-xv0*(t-t0))**2+(yv1*(t-t1)-yv0*(t-t0))**2+(zv1*(t-t1)-zv0*(t-t0))**2=r**2
                    #(x1-x0)**2+(y1-y0)**2+(z1-z0)**2+2*(((x1-x0)*xv1*(t-t1)+(y1-y0)*yv1*(t-t1)+(z1-z0)*zv1*(t-t1))-((x1-x0)*xv0*(t-t0)+(y1-y0)*yv0*(t-t0)+(z1-z0)*zv0*(t-t0)))+(xv1*(t-t1)-xv0*(t-t0))**2+(yv1*(t-t1)-yv0*(t-t0))**2+(zv1*(t-t1)-zv0*(t-t0))**2=r**2
                    #(x1-x0)**2+(y1-y0)**2+(z1-z0)**2+2*(((x1-x0)*xv1+(y1-y0)*yv1+(z1-z0)*zv1)*(t-t1)-((x1-x0)*xv0+(y1-y0)*yv0+(z1-z0)*zv0)*(t-t0))+(xv1*(t-t1)-xv0*(t-t0))**2+(yv1*(t-t1)-yv0*(t-t0))**2+(zv1*(t-t1)-zv0*(t-t0))**2=r**2
                    #(x1-x0)**2+(y1-y0)**2+(z1-z0)**2+2*(((x1-x0)*xv1+(y1-y0)*yv1+(z1-z0)*zv1)*(t-t1)-((x1-x0)*xv0+(y1-y0)*yv0+(z1-z0)*zv0)*(t-t0))+xv1**2*(t-t1)**2-2*xv1*(t-t1)*xv0*(t-t0)+xv0**2*(t-t0)**2+yv1**2*(t-t1)**2-2*yv1*(t-t1)*yv0*(t-t0)+yv0**2*(t-t0)**2+zv1**2*(t-t1)**2-2*zv1*(t-t1)*zv0*(t-t0)+zv0**2*(t-t0)**2=r**2
                    #(x1-x0)**2+(y1-y0)**2+(z1-z0)**2+2*(((x1-x0)*xv1+(y1-y0)*yv1+(z1-z0)*zv1)*(t-t1)-((x1-x0)*xv0+(y1-y0)*yv0+(z1-z0)*zv0)*(t-t0))+(xv1**2+yv1**2+zv1**2)*(t-t1)**2-2*(xv1+yv1+zv1)*(t-t1)*(xv0+yv0+zv0)*(t-t0)+(xv0**2+yv0**2+zv0**2)*(t-t0)**2-r**2=0
                    #(x1-x0)**2+(y1-y0)**2+(z1-z0)**2+(xv1**2+yv1**2+zv1**2)*(t-t1)**2+2*((x1-x0)*xv1+(y1-y0)*yv1+(z1-z0)*zv1)*(t-t1)-2*(xv1+yv1+zv1)*(t-t1)*(xv0+yv0+zv0)*(t-t0)+(xv0**2+yv0**2+zv0**2)*(t-t0)**2-2*((x1-x0)*xv0+(y1-y0)*yv0+(z1-z0)*zv0)*(t-t0)-r**2=0
                    #Mathematica says
                    #t=-((x0*xv0-x1*xv0-x0*xv1+x1*xv1+y0*yv0-y1*yv0-y0*yv1+y1*yv1+z0*zv0-z1*zv0-z0*zv1+z1*zv1-xv0**2*t0+xv0*xv1*t0+xv1*yv0*t0-yv0**2*t0+xv0*yv1*t0+yv0*yv1*t0+xv1*zv0*t0+yv1*zv0*t0-zv0**2*t0+xv0*zv1*t0+yv0*zv1*t0+zv0*zv1*t0+xv0*xv1*t1-xv1**2*t1+xv1*yv0*t1+xv0*yv1*t1+yv0*yv1*t1-yv1**2*t1+xv1*zv0*t1+yv1*zv0*t1+xv0*zv1*t1+yv0*zv1*t1+zv0*zv1*t1-zv1**2*t1±math.sqrt(4*(x0*(xv0-xv1)+x1*(xv1-xv0)+y0*yv0-y1*yv0-y0*yv1+y1*yv1+z0*zv0-z1*zv0-z0*zv1+z1*zv1-xv0**2*t0+xv0*xv1*t0+xv1*yv0*t0-yv0**2*t0+xv0*yv1*t0+yv0*yv1*t0+xv1*zv0*t0+yv1*zv0*t0-zv0**2*t0+xv0*zv1*t0+yv0*zv1*t0+zv0*zv1*t0+xv0*xv1*t1-xv1**2*t1+xv1*yv0*t1+xv0*yv1*t1+yv0*yv1*t1-yv1**2*t1+xv1*zv0*t1+yv1*zv0*t1+xv0*zv1*t1+yv0*zv1*t1+zv0*zv1*t1-zv1**2*t1)**2-4*(xv0**2+xv1**2+yv0**2-2*yv0*yv1+yv1**2-2*yv1*zv0+zv0**2-2*xv1*(yv0+zv0)-2*yv0*zv1-2*zv0*zv1+zv1**2-2*xv0*(xv1+yv1+zv1))*(x0**2+x1**2+y0**2-2*y0*y1+y1**2+z0**2-2*z0*z1+z1**2-r**2+2*x1*xv0*t0-2*y0*yv0*t0+2*y1*yv0*t0-2*z0*zv0*t0+2*z1*zv0*t0+xv0**2*t0**2+yv0**2*t0**2+zv0**2*t0**2-2*x1*xv1*t1+2*y0*yv1*t1-2*y1*yv1*t1+2*z0*zv1*t1-2*z1*zv1*t1-2*xv0*xv1*t0*t1-2*xv1*yv0*t0*t1-2*xv0*yv1*t0*t1-2*yv0*yv1*t0*t1-2*xv1*zv0*t0*t1-2*yv1*zv0*t0*t1-2*xv0*zv1*t0*t1-2*yv0*zv1*t0*t1-2*zv0*zv1*t0*t1+xv1**2*t1**2+yv1**2*t1**2+zv1**2*t1**2-2*x0*(x1+xv0*t0-xv1*t1)))/2)/(xv0**2+xv1**2+yv0**2-2*yv0*yv1+yv1**2-2*yv1*zv0+zv0**2-2*xv1*(yv0+zv0)-2*yv0*zv1-2*zv0*zv1+zv1**2-2*xv0*(xv1+yv1+zv1))) #thank you Stephen Wolfram
                    #simplifies to
                    #t=-(((x0-x1)*(xv0-xv1)+(y0-y1)*(yv0-yv1)+(z0-z1)*(zv0-zv1)-(xv0**2+yv0**2+zv0**2)*t0-(xv1**2+yv1**2+zv1**2)*t1+(xv0+yv0+zv0)*(xv1+yv1+zv1)*(t0+t1)±math.sqrt(((x0-x1)*(xv0-xv1)+(y0-y1)*(yv0-yv1)+(z0-z1)*(zv0-zv1)-(xv0**2+yv0**2+zv0**2)*t0-(xv1**2+yv1**2+zv1**2)*t1+(xv0+yv0+zv0)*(xv1+yv1+zv1)*(t0+t1))**2-(xv0**2+xv1**2+yv0**2+yv1**2+zv0**2+zv1**2-2*(xv0+yv0+zv0)*(xv1+yv1+zv1))*(x0**2+x1**2+y0**2+y1**2+z0**2+z1**2-r**2+2*((x0-x1)*(xv1*t1-xv0*t0)-x0*x1+(y0-y1)*(yv1*t1-yv0*t0)-y0*y1+(z0-z1)*(zv1*t1-zv0*t0)-z0*z1)+(xv0**2+yv0**2+zv0**2)*t0**2-2*(xv0+yv0+zv0)*(xv1+yv1+zv1)*t0*t1+(xv1**2+yv1**2+zv1**2)*t1**2)))/(xv0**2+xv1**2+yv0**2+yv1**2+zv0**2+zv1**2-2*(xv0+yv0+zv0)*(xv1+yv1+zv1)))
                    #t=-(((x0-x1)*(xv0-xv1)+(y0-y1)*(yv0-yv1)+(z0-z1)*(zv0-zv1)-(xv0**2+yv0**2+zv0**2)*t0-(xv1**2+yv1**2+zv1**2)*t1+(xv0+yv0+zv0)*(xv1+yv1+zv1)*(t0+t1)±math.sqrt(((x0-x1)*(xv0-xv1)+(y0-y1)*(yv0-yv1)+(z0-z1)*(zv0-zv1)-(xv0**2+yv0**2+zv0**2)*t0-(xv1**2+yv1**2+zv1**2)*t1+(xv0+yv0+zv0)*(xv1+yv1+zv1)*(t0+t1))**2-(xv0**2+xv1**2+yv0**2+yv1**2+zv0**2+zv1**2-2*(xv0+yv0+zv0)*(xv1+yv1+zv1))*((x0-x1)**2+(y0-y1)**2+(z0-z1)**2-r**2+2*((x0-x1)*(xv1*t1-xv0*t0)+(y0-y1)*(yv1*t1-yv0*t0)+(z0-z1)*(zv1*t1-zv0*t0))+(xv0**2+yv0**2+zv0**2)*t0**2-2*(xv0+yv0+zv0)*(xv1+yv1+zv1)*t0*t1+(xv1**2+yv1**2+zv1**2)*t1**2)))/(xv0**2+xv1**2+yv0**2+yv1**2+zv0**2+zv1**2-2*(xv0+yv0+zv0)*(xv1+yv1+zv1)))
                    #better computation
                    #b=(x0-x1)*(xv0-xv1)+(y0-y1)*(yv0-yv1)+(z0-z1)*(zv0-zv1)-(xv0**2+yv0**2+zv0**2)*t0-(xv1**2+yv1**2+zv1**2)*t1+(xv0+yv0+zv0)*(xv1+yv1+zv1)*(t0+t1)
                    #a=xv0**2+xv1**2+yv0**2+yv1**2+zv0**2+zv1**2-2*(xv0+yv0+zv0)*(xv1+yv1+zv1)
                    #discriminant=bee**2-a*((x0-x1)**2+(y0-y1)**2+(z0-z1)**2-r**2+2*((x0-x1)*(xv1*t1-xv0*t0)+(y0-y1)*(yv1*t1-yv0*t0)+(z0-z1)*(zv1*t1-zv0*t0))+(xv0**2+yv0**2+zv0**2)*t0**2+(xv1**2+yv1**2+zv1**2)*t1**2-2*(xv0+yv0+zv0)*(xv1+yv1+zv1)*t0*t1)
                    #t=(-b±math.sqrt(discriminant))/a
                    bee=sum((mi-ni)*(mo-no) for mi,mo,ni,no in zip(k[0][0],k[0][1],l[0][0],l[0][1]))-sum(m**2 for m in k[0][1])*t-sum(n**2 for n in l[0][1])*u+sum(k[0][1])*sum(l[0][1])*(t+u)
                    #a=xv0**2+xv1**2+yv0**2+yv1**2+zv0**2+zv1**2-2*(xv0+yv0+zv0)*(xv1+yv1+zv1) #equals (xv0-xv1)**2+(yv0-yv1)**2+(zv0-zv1)**2 (the magnitude of the difference in velocity) plus something like the cross product but with only positive coefficients and returning the sum of axes like the dot product does (I call it the dross product)
                    a=sum(m**2 for m in k[0][1])+sum(n**2 for n in l[0][1])-2*sum(k[0][1])*sum(l[0][1])
                    radius=k[1][0]+l[1][0]
                    #discriminant=bee**2-a*((x0-x1)**2+(y0-y1)**2+(z0-z1)**2-r**2+2*((x0-x1)*(xv1*t1-xv0*t0)+(y0-y1)*(yv1*t1-yv0*t0)+(z0-z1)*(zv1*t1-zv0*t0))+(xv0**2+yv0**2+zv0**2)*t0**2+(xv1**2+yv1**2+zv1**2)*t1**2-2*(xv0+yv0+zv0)*(xv1+yv1+zv1)*t0*t1)
                    #discriminant=bee**2-a*(sum((mi-ni)**2+2*(mi-ni)*(mo*t-no*u)+(mo*t)**2+(no*u)**2 for mi,mo,ni,no in zip(k[0][0],k[0][1],l[0][0],l[0][1]))-radius**2-2*sum(k[0][1])*sum(l[0][1])*t*u)
                    discriminant=bee**2-a*(sum(((mi-ni)+(mo*t-no*u))**2 for mi,mo,ni,no in zip(k[0][0],k[0][1],l[0][0],l[0][1]))-radius**2-2*drossProduct(k[0][1],l[0][1])*t*u) #once again, the dross product (what does it all mean)
                    if discriminant>0:
                        discriminant=math.sqrt(discriminant)
                        #times=[(-bee+p*discriminant)/a for p in range(-1,3,2)] #all coefficients cancel out if you think about it
                        candidate=(-bee-discriminant)/a #times[times[0]<0] #no longer account for exiting spheres
                        if 0<candidate<bestCandidateTime:
                            bestCandidateTime=candidate
                            bestCandidateTimes=(t,u)
                            bestCandidateRadius=radius
                            bestCandidateNodes=(k,l)
                            bestCandidateIndices=(i,j)
                    else:
                        pairsAlreadyCollided.append((i,j))
        if armExtended and nodes[-1][4]==3:
            nodes[-1][0][1]=nodes[nodes[-1][5]][0][1]
            if math.dist(nodes[-1][0][0],nodes[0][0][0])>=nodes[-1][6]:
                displacement=lap(float.__sub__,nodes[0][0][0],nodes[-1][0][0])
                normalDisplacement=[d/nodes[-1][6] for d in displacement]
                dot=dotProduct(map(float.__sub__,nodes[0][0][1],nodes[-1][0][1]),normalDisplacement)
                if dot+contractionRate>0:
                    nodes[0][0]=[[n+nd*nodes[-1][6]-d for n,d,nd in zip(nodes[0][0][0],displacement,normalDisplacement)],[(v-n*dot)/contractionRatio-n*contractionRate for v,n in zip(nodes[0][0][1],normalDisplacement)]]
                #pairsAlreadyCollided=[(i,j) for i,j in pairsAlreadyCollided if i!=0!=j]
        '''for i in nodes:
            i[0][0]=[s+v*bestCandidateTime for (s,v) in zip(*i[0])] #formerly lap(float.__add__,*i[0]) (back in my day before we had to bother with all of this sub-frame nonsense)'''
        if bestCandidateIndices!=(-1,)*2:
            for i,j,t in zip(bestCandidateIndices,bestCandidateNodes,bestCandidateTimes):
                nodes[i][0][0]=[s+v*(bestCandidateTime-t) for s,v in zip(*j[0])]
                t=bestCandidateTime
        timeElapsed+=bestCandidateTime
        if timeElapsed<1:
            #print(timeElapsed)
            #if bestCandidateIndices[0]==0 and bestCandidateNodes[1][4]==2:
                #nodes[bestCandidateIndices[1]][4]+=1 #arm formerly 'collided' with player when it stopped intersecting
            itsOver=False
            if bestCandidateIndices[0]==0 and bestCandidateNodes[1][4]==2:
                deleteArm(bestCandidateIndices[1])
            else:
                if bestCandidateNodes[1][4]==2:
                    nodes[bestCandidateIndices[1]][0][1]=bestCandidateNodes[1][0][1]
                    nodes[bestCandidateIndices[1]][4]=3
                    nodes[bestCandidateIndices[1]].append(bestCandidateIndices[0]) #index of node it is connected to
                    nodes[bestCandidateIndices[1]].append(math.dist(bestCandidateNodes[1][0][0],nodes[0][0][0])) #length of arm from player
                elif bestCandidateNodes[1][4]==3:
                    if bestCandidateIndices[0]==0:
                        deleteArm(bestCandidateIndices[1])
                    #else:
                        #moved to be unconditional instead
                elif bestCandidateNodes[1][4]==1: #sorry for inelegance here
                    nodes[bestCandidateIndices[0]][0][1]=tangentCollision(bestCandidateNodes[1],bestCandidateNodes[0][0],bestCandidateRadius,True,1,False,(gameMode and bestCandidateIndices[0]==0))
                    if gameMode and bestCandidateIndices[0]==0:
                        playerLanded=True
                        tangentDivergence=math.hypot(*map(float.__sub__,map(float.__add__,*nodes[bestCandidateIndices[0]][0]),nodes[bestCandidateIndices[1]][0][0]))-(nodes[bestCandidateIndices[0]][1][0]+nodes[bestCandidateIndices[1]][1][0])
                        if armExtended:
                            deleteArm(-1)
                elif bestCandidateNodes[0][4]==1:
                    nodes[bestCandidateIndices[1]][0][1]=tangentCollision(bestCandidateNodes[0],bestCandidateNodes[1][0],bestCandidateRadius,True,1,False)
                else:
                    for i,v in zip(bestCandidateIndices,tangentCollision(*bestCandidateNodes,bestCandidateRadius)):
                        nodes[i][0][1]=v
            pairsAlreadyCollided=[(i,j) for i,j in pairsAlreadyCollided if i not in bestCandidateIndices and j not in bestCandidateIndices or 2<=nodes[j][4]<4]+[bestCandidateIndices]*(not itsOver) #we're all going to make it
    for i,(j,t) in enumerate(zip(nodes,nodeFramesProceeded)):
        nodes[i][0][0]=[s+v*(1-t) for s,v in zip(*j[0])]
def quaternionMultiply(a,b):
    return ( (a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
              a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
              a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
              a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]) #this function not to be taken to before 1843
            if dims==3 else
             (a[0]*b[0]-a[1]*b[1],
              a[0]*b[1]+a[1]*b[0])) #this one is fine

def quaternionConjugate(q): #usually conjugation is inverting the imaginary parts but as all quaternion enthusiasts know, inverting all four axes is ineffectual so inverting the first is ineffectually different from inverting the other three
    return (-q[0],)+q[1:4] if dims==3 else (q[0],-q[1]) #(if you care about memory and not computing power, only the other three need to be stored with their polarities relative to the real component, which can have its sign fixed and its magnitude computed (because it's a unit vector))
#the reciprocal is the conjugate for unit vectors, otherwise divide by the magnitude squared

def rotateVector(v,q):
    return ( ((2*(q[2]**2+q[3]**2)-1)*v[0]+2*((q[0]*q[3]-q[1]*q[2])*v[1]-(q[0]*q[2]+q[1]*q[3])*v[2]),
              (2*(q[1]**2+q[3]**2)-1)*v[1]+2*((q[0]*q[1]-q[2]*q[3])*v[2]-(q[1]*q[2]+q[0]*q[3])*v[0]),
              (2*(q[1]**2+q[2]**2)-1)*v[2]+2*((q[0]*q[2]-q[1]*q[3])*v[0]-(q[0]*q[1]+q[2]*q[3])*v[1]))
            if dims==3 else
             (q[0]*v[0]-q[1]*v[1], #q means qomplex number instead of quaternion
              q[1]*v[0]+q[0]*v[1]))
def antiReferenceFrame(v):
    return rotateVector(v,quaternionConjugate(cameraAngle[0]))

def rotationMatrix(q): #convert quaternion to rotation matrix #like rotateVector but returning matrix of coefficients for arbitrary vectors instead
    return (( 2*(q[2]**2  +q[3]**2)-1, 2*(q[0]*q[3]-q[1]*q[2]),-2*(q[0]*q[2]+q[1]*q[3]),
             -2*(q[1]*q[2]+q[0]*q[3]), 2*(q[1]**2  +q[3]**2)-1, 2*(q[0]*q[1]-q[2]*q[3]),
              2*(q[0]*q[2]-q[1]*q[3]),-2*(q[0]*q[1]+q[2]*q[3]), 2*(q[1]**2  +q[2]**2)-1)
            if dims==3 else
             (q[0],-q[1],
              q[1], q[0]))
def rotateByMatrix(v,m):
    return ((m[0]*v[0]+m[1]*v[1]+m[2]*v[2],
             m[3]*v[0]+m[4]*v[1]+m[5]*v[2],
             m[6]*v[0]+m[7]*v[1]+m[8]*v[2])
            if dims==3 else
             (m[0]*v[0]+m[1]*v[1],
              m[2]*v[0]+m[3]*v[1])
            if dims==2 else
            #tuple(sum(m[di*dims+i]*vi for i,vi in enumerate(v)) for di in range(dims))
             tuple(sum(map(mul,m[di*dims:(di+1)*dims],v)) for di in range(dims)))
def matrixMultiply(a,b):
    return ( (a[0]*b[0]+a[1]*b[3]+a[2]*b[6],a[0]*b[1]+a[1]*b[4]+a[2]*b[7],a[0]*b[2]+a[1]*b[5]+a[2]*b[8],
              a[3]*b[0]+a[4]*b[3]+a[5]*b[6],a[3]*b[1]+a[4]*b[4]+a[5]*b[7],a[3]*b[2]+a[4]*b[5]+a[5]*b[8],
              a[6]*b[0]+a[7]*b[3]+a[8]*b[6],a[6]*b[1]+a[7]*b[4]+a[8]*b[7],a[6]*b[2]+a[7]*b[5]+a[8]*b[8])
            if dims==3 else
             (a[0]*b[0]+a[1]*b[2],a[0]*b[1]+a[1]*b[3],
              a[2]*b[0]+a[3]*b[2],a[2]*b[1]+a[3]*b[3])
            if dims==2 else
            #tuple(sum(a[i*dims+k]*b[j+k*dims] for k in range(dims)) for i in range(dims) for j in range(dims)))
             tuple(sum(map(float.__mul__,a[i*dims:(i+1)*dims],b[j:j+dims**2:dims])) for i in range(dims) for j in range(dims))) #I like the most
            #tuple(sum(map(float.__mul__,a[i//dims*dims:(i//dims+1)*dims],b[i%dims:i%dims+dims**2:dims])) for i in range(dims**2))

def quaternion(m): #convert rotation matrix to quaternion #based on https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf, which is based on http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/, but my convention seems to be negated and transposed so you must take it down, flip it and reverse it
    #m[5]-m[7]       =2*(q[0]*q[1]-q[2]*q[3])+2*(q[0]*q[1]+q[2]*q[3])                        =4*q[1]*q[0]
    #1-m[0]+m[4]+m[8]=1-(2*(q[2]**2+q[3]**2)-1)+(2*(q[1]**2+q[3]**2)-1+2*(q[1]**2+q[2]**2)-1)=4*q[1]**2
    #-m[1]-m[3]      =-2*(q[0]*q[3]-q[1]*q[2])+2*(q[1]*q[2]+q[0]*q[3])                       =4*q[1]*q[2]
    #-m[2]-m[6]      =2*(q[0]*q[2]+q[1]*q[3])-2*(q[0]*q[2]-q[1]*q[3])                        =4*q[1]*q[3]
    #purificationFactor=1/(2*math.sqrt(1-m[0]+m[4]+m[8]))
    if m[8]>0:
        if m[0]<m[4]:
            t=-m[0]+m[4]+m[8]-1
            q=[t,-m[1]-m[3],-m[2]-m[6],m[7]-m[5]]
        else:
            t= m[0]-m[4]+m[8]-1
            q=[-m[1]-m[3],t,-m[5]-m[7],m[6]-m[2]]
    else:
        if m[0]>-m[4]:
            t= m[0]+m[4]-m[8]-1
            q=[-m[2]-m[6],-m[5]-m[7],t,m[3]-m[1]]
        else:
            t= m[0]+m[4]+m[8]-1
            q=[m[7]-m[5],m[2]-m[6],m[3]-m[1],t]
    t=t**-0.5/2
    return [i/t for i in q]

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
    coeff=math.sin(theSquares)*sgn(qzero)/math.sqrt(1-qzero**2) #could instead be divided by ((a[0]*b[1]-a[1]*b[0]-a[2]*b[3]+a[3]*b[2])**2+(a[0]*b[2]+a[1]*b[3]-a[2]*b[0]-a[3]*b[1])**2+(a[0]*b[3]-a[1]*b[2]+a[2]*b[1]-a[3]*b[0])**2)
    #sgn(qzero)=qzero/sqrt(qzero**2), qzero/sqrt(qzero**2)*sqrt(1-qzero**2)=qzero*sqrt((1-qzero**2)/qzero**2)=x*sqrt(qzero**-2-1)
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
#print([slerp([1,0,0,0],[1/3,-1/6,math.sqrt(3)/2,-1/3],n/4) for n in range(5)])
#print([math.hypot(*slerp([1,0,0,0],[1/3,-1/6,math.sqrt(3)/2,-1/3],n/100)) for n in range(100)]) #test that it yields intermediate unit vectors
#print([slerp([1,0,0,0],[-1/3,-1/6,math.sqrt(3)/2,-1/3],n/4) for n in range(5)]) #test that it takes optimal route

def slerpMatrices(a,b,t): #I thought there would be a closed form but it seems there isn't
    return rotationMatrix(slerp(quaternion(a),quaternion(b),t))

pixelAngle=1/minSize #math.tau/max(size)
def rotateByScreen(angle,screenRotation):
    magnitude=math.hypot(*screenRotation)
    if magnitude==0:
        return angle
    else:
        magpi=magnitude*pixelAngle
        simagomag=math.sin(magpi)/magnitude #come and get your simagomags delivered fresh
        return quaternionMultiply([math.cos(magpi)]+[i*simagomag for i in screenRotation],angle)

projectionMode=2*(dims==3)
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
    #antiProject(x,y,cameraAngle[0]) is equivalent to rotateVector((0,1,0),rotateByScreen(cameraAngle[0],(x,y,0)))
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
    #rotateVector((0,1,0),q) yields
    #return [ 2*((q[0]*q[3]-q[1]*q[2])),
            #(2*(q[1]**2+q[3]**2)-1),
            #-2*(q[0]*q[1]+q[2]*q[3])]
    return [2*(((comag*q[0]-simagomag*(x*q[1]+y*q[2]))*(comag*q[3]+simagomag*(x*q[2]-y*q[1]))-(comag*q[1]+simagomag*(x*q[0]+y*q[3]))*(comag*q[2]-simagomag*(x*q[3]-y*q[0])))),
            2*((comag*q[1]+simagomag*(x*q[0]+y*q[3]))**2+(comag*q[3]+simagomag*(x*q[2]-y*q[1]))**2)-1,
           -2*((comag*q[0]-simagomag*(x*q[1]+y*q[2]))*(comag*q[1]+simagomag*(x*q[0]+y*q[3]))+(comag*q[2]-simagomag*(x*q[3]-y*q[0]))*(comag*q[3]+simagomag*(x*q[2]-y*q[1])))]

def rayAngle(x,y): #converts screen position to ray velocity vector
    if projectionMode==0:
        raise RuntimeError("the weird projection isn't supported for raytracing") #too weird
    else:
        #x-=halfSize[0]
        #y-=halfSize[1]
        magnitude=pixelAngle*math.hypot(x,y)
        #print(magnitude/math.sqrt(2))
        bagnitude=(magnitude if projectionMode==1 else 2*math.asin(magnitude/math.sqrt(2)) if projectionMode==2 else 2*math.atan(1/magnitude)/magnitude) #if projectionMode==3 else)
        #Wikipedia says use 2*math.asin(magnitude/math.sqrt(2)) but it doesn't seem to work (I will instead use the one from the Scratch one)
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
            if False: #for testing
                rayPosition=rayAngle((screenJ-halfSize[0])/fieldOfView[0],(screenI-halfSize[1])/fieldOfView[1])
                antialiasColour=angleColour(math.pi*((rayPosition[0]<0) if abs(rayPosition[1])<abs(rayPosition[0])>abs(rayPosition[2]) else 1/3+(rayPosition[1]<0) if abs(rayPosition[1])>abs(rayPosition[2]) else 2/3+(rayPosition[2]<0)))
            else:
                antialiasColour=[0,0,0]
                for antialiasPasses in range(antialiasing if antialiasing else 1):
                    timeElapsed=0
                    (i,j,t)=[random.random()-0.5 for _ in range(2+timeAntialiasing)]+[0]*(not timeAntialiasing) if antialiasing else (0,)*3
                    i=screenI+i-halfSize[(dims==3)]
                    j=screenJ+j-halfSize[(dims==2)]
                    if dims==2:
                        '''p=lap(float.__add__,rotateByMatrix([(n-c-h)*orthographicZoom for n,c,h in zip(n[0][0],cameraPosition[0],halfSize)],cameraMatrix),halfSize)
                        lap(float.__sub__,p,halfSize)=rotateByMatrix([(n-c-h)*orthographicZoom for n,c,h in zip(n[0][0],cameraPosition[0],halfSize)],cameraMatrix)
                        antiReferenceFrame(lap(float.__sub__,p,halfSize))=[(n-c-h)*orthographicZoom for n,c,h in zip(n[0][0],cameraPosition[0],halfSize)]
                        [o/orthographicZoom for o in antiReferenceFrame(lap(float.__sub__,p,halfSize))]=[n-c-h for n,c,h in zip(n[0][0],cameraPosition[0],halfSize)]
                        [o/orthographicZoom+c+h for o,c,h in zip(antiReferenceFrame(lap(float.__sub__,p,halfSize)),cameraPosition[0],halfSize)]=n''' #from there I kept moving things around until it was calibrated
                        rayPosition=[o+h+c for o,c,h in zip(antiReferenceFrame((i/orthographicZoom,j/orthographicZoom)),cameraPosition[0],halfSize)]
                        gravityAcceleration=[0]*2
                        for m,n in enumerate(nodes):
                                differences=[s-r for s,r in zip(n[0][0],rayPosition)]
                                gravity=gravitationalConstant/max(2,sum(di**2 for di in differences)**1.5)*n[2]
                                gravityAcceleration=[gi+gravity*di for gi,di in zip(gravityAcceleration,differences)]
                        rayColour=angleColour(math.atan2(*gravityAcceleration))
                    elif dims==3:
                        rayPosition=[cameraPosition[0],rayAngle(j/fieldOfView[0],i/fieldOfView[1])]
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
                                (mag,newVelocity)=tangentCollision(bestCandidateNode,rayPosition,bestCandidateNode[1][1],True) #mag being collision magnitude
                                if normalise:
                                    #speed=math.hypot(*newVelocity)
                                    newVelocity=[v*lightSpeed/mag for v in newVelocity]
                                #mag=math.hypot(*(n-o for n,o in zip(newVelocity,rayPosition[1])))
                                rayPosition[1]=newVelocity
                                reflectivenesses.append(1-(1-bestCandidateNode[3][3]*mag/(lightSpeed if lightSpeed else 1)) if True else bestCandidateNode[3][3])
                                reflectionColours.append(bestCandidateNode[3][:3])
                            reflections+=1
                        rayColour=[0,0,0]
                        for r,c in zip(reflectivenesses[::-1],reflectionColours[::-1]):
                            rayColour=weightedAverageColours((1-r,rayColour),(r,c))
                antialiasColour=(weightedAverageColours((antialiasPasses/(antialiasPasses+1),antialiasColour),(1/(antialiasPasses+1),rayColour)) if antialiasing else rayColour)
                antialiasColour=[min(round(i*exposure),255) for i in antialiasColour]
                '''if antialiasColour!=[0]*3:
                    print("radius",pixelRadius,"position",(screenI/(upscale)/(fieldOfView[1]/pixelAngle/jInc/upscale)*size[0],hexLattice/theThing-screenJ/(jInc*upscale)/((fieldOfView[0]/pixelAngle+jInc*hexLattice*screenK%2)/upscale)*size[1]),"colour",antialiasColour)
                    print("(",screenI,"*",size[0],"/",fieldOfView[0],",",hexLattice,"/",theThing,"-",screenJ,"*",size[1],"/",fieldOfView[1],")")'''
            #drawShape((pixelRadius if hexLattice else (pixelRadius,)*2),((screenI/upscale-2/2)/(fieldOfView[0]/pixelAngle/jInc/upscale)*size[0],hexLattice/theThing+(screenJ+1/2)/(jInc*upscale)/((fieldOfView[1]/pixelAngle+jInc*hexLattice*screenK%2)/upscale)*size[1]),antialiasColour,3*hexLattice)
            drawShape((pixelRadius if hexLattice else (pixelRadius,)*2),(screenI*size[0]/(fieldOfView[0]/pixelAngle),(screenJ-hexLattice/theThing)*size[1]/((fieldOfView[1]/pixelAngle))),antialiasColour,3*hexLattice)

def findNodeScreenPositions():
    return ( [projectRelativeToScreen(rotateByMatrix(lap(float.__sub__,n[0][0],cameraPosition[0]),cameraMatrix)) for n in nodes]
            if dims==3 and perspectiveMode else
             [lap(float.__add__,rotateByMatrix([(n-c-h)*orthographicZoom for n,c,h in zip(n[0][0],cameraPosition[0],halfSize)],cameraMatrix),halfSize) for n in nodes]) #list comprehension because would have to be lap(float.__sub__,lap(float.__sub__,n[0][0],cameraPosition[0]),halfSize)) because __sub__ doesn't take arbitrarily many parameters

angularGain=math.tau/FPS
perspectiveMode=True
physicsMode=True
raytracingMode=False
toggleKeys=[pygame.K_z,pygame.K_t,pygame.K_MINUS,pygame.K_EQUALS]
oldToggles=[False]*len(toggleKeys)
run=True
def average(*vectors):
    l=len(vectors)
    return [sum(i)/l for i in zip(*vectors)]

frames=0
framesMouseDown=0
mouseChangeSinceClick=[0,0]
spaceChangeSinceClick=[0.0,0.0]
playerLanded=False
armExtended=False
tangentDivergence=0.0
contractionRate=0.0
contractionRatio=1.0
while run:
    keys=pygame.key.get_pressed()
    doEvents()
    if clickDone:
        mousePos=[m-h for m,h in zip(mouse.get_pos(),halfSize)] #not lap(int.__sub__,mouse.get_pos(),halfSize) because halfSize is floats (my macOS maximum non-fullscreen window height is odd, you see)
    if perspectiveMode:
        cameraPosition=[((lap(float.__add__,nodes[0][0][0],antiReferenceFrame(playerNodeOffset)) if dims==3 else lap(float.__sub__,nodes[0][0][0],halfSize)) if playerNode and physicsMode else lap(float.__add__,*cameraPosition)),([0.0]*3 if keys[pygame.K_LSHIFT] else nodes[0][0][1] if gameMode and not playerLanded else lap(float.__add__,(nodes[0][0][1] if playerNode and physicsMode else cameraPosition[1]),antiReferenceFrame([(keys[pygame.K_d]-keys[pygame.K_a])*spatialGain,tangentDivergence] if gameMode and physicsMode else [k*spatialGain for k in (normalise([keys[pygame.K_d]-keys[pygame.K_a],keys[pygame.K_f]-keys[pygame.K_r],keys[pygame.K_w]-keys[pygame.K_s]] if dims==3 else ([keys[pygame.K_d]-keys[pygame.K_a],keys[pygame.K_s]-keys[pygame.K_w]])))])))]
        if gameMode:
            if shortClickDone:
                if playerLanded:
                    cameraPosition[1]=lap(float.__add__,nodes[0][0][1],antiReferenceFrame((0.0,spatialGain*-32.0)))
                elif armExtended:
                    del nodes[-1]
                    del stateTransitions[-1]
                    armExtended=False
                else:
                    nodes.append([[nodes[0][0][0],lap(float.__add__,nodes[0][0][1],normalise(antiReferenceFrame(mousePos),False,spatialGain*32))],(rad/8,)*dims,(1/8)**dims,(255,)*3,2])
                    stateTransitions.append([])
                    armExtended=True
                    contractionRate=0.0
                    contractionRatio=1.0
            elif nodes[-1][4]==3 and mouse.get_pressed()[0] and armExtended:
                contractionRate=(spatialGain*8 if nodes[-1][6]>spatialGain*24 else nodes[-1][6]-spatialGain*16)
                contractionRatio=1-contractionRate/nodes[-1][6]
                nodes[-1][6]-=contractionRate
            else:
                contractionRate=0.0
        nodes[0][0][1]=cameraPosition[1]
    else:
        cameraPosition[0]=averageNode()
    toggles=[keys[k] for k in toggleKeys]
    for i,(k,o) in enumerate(zip(toggles,oldToggles)):
        if not k and o:
            if i==0: #space (between turn to move, winningness and DTM)
                physicsMode^=True #(because it can be in real time without O(n*(n-1)/2) gravity simulation)
            elif i==1:
                raytracingMode^=True
            elif i==2:
                orthographicZoom/=math.sqrt(2)
            elif i==3:
                orthographicZoom*=math.sqrt(2)
    oldToggles=toggles
    if not gameMode:
        if mouse.get_pressed()[0]:
            mouseChange=mouse.get_rel()
            if dims==3:
                mouseChange=(-mouseChange[1],mouseChange[0],0) #mouseChange+=(0,)
                mouseChangeSinceClick=lap(float.__add__,mouseChangeSinceClick,lap(float,mouseChange))
                wibble=lap(float.__sub__,mouseChangeSinceClick,spaceChangeSinceClick)
            else:
                if not (playerNode and physicsMode):
                    cameraPosition[0]=lap(float.__add__,cameraPosition[0],(v/orthographicZoom for v in antiReferenceFrame(lap(int.__neg__,mouseChange)))) #the dims==3 here will never happen but is for dragging in the screen axes
        else:
            mouse.get_rel() #otherwise it jumps
            mouseChange=(0,)*3
            mouseChangeSinceClick=[0.0,0.0,0.0]
            spaceChangeSinceClick=[0.0,0.0,0.0]
    keyAcceleration=([keys[pygame.K_DOWN]-keys[pygame.K_UP],keys[pygame.K_LEFT]-keys[pygame.K_RIGHT],keys[pygame.K_q]-keys[pygame.K_e]] if dims==3 else [keys[pygame.K_q]-keys[pygame.K_e]])
    angularAcceleration=lap(float.__add__,map(float,keyAcceleration),[mouseChange[1]/minSize,mouseChange[0]/minSize,0]) if dims==3 else keyAcceleration
    #v**2=u**2+2*a*s (stay in school kids)
    #0**2<=(dotProduct(wibble,cameraAngle[1])/abs(wibble))**2+2*angularGain*abs(wibble)
    #-(dotProduct(arg(wibble),cameraAngle[1]))**2<=2*angularGain*abs(wibble)
    #-(dotProduct(wibble,cameraAngle[1])/math.hypot(*wibble))**2<=2*angularGain*math.hypot(*wibble)
    #-dotProduct(wibble,cameraAngle[1])**2<=2*angularGain*math.hypot(*wibble)**3
    if not gameMode:
        cameraAngle=[rotateByScreen(cameraAngle[0],mouseChange),[0.0]*3] if keys[pygame.K_LSHIFT] and drag==0 else [rotateByScreen(*cameraAngle),[(di+acc*angularGain)/(1+drag) for di,acc in zip(cameraAngle[1],normalise(((wibble if (dims==3 and mouse.get_pressed()[0]) and dotProduct(wibble,cameraAngle[1])**2<2*angularGain*math.hypot(*wibble)**3 else lap(float.__neg__,cameraAngle[1])) if keyAcceleration==[0,0,0] else angularAcceleration),keyAcceleration==[0.0,0.0,0.0]))]]
    if mouse.get_pressed()[0]:
        spaceChangeSinceClick=lap(float.__add__,spaceChangeSinceClick,cameraAngle[1])
        #print(mouseChangeSinceClick,spaceChangeSinceClick)
    cameraMatrix=rotationMatrix(cameraAngle[0])
    nodeScreenPositions=findNodeScreenPositions()
    if physicsMode:
        physics()
        subframes()
    renderOrder=range(len(nodes)) if dims==2 else conditionalReverse(perspectiveMode,[j for _, j in sorted((p[2],i) for i,p in enumerate(nodeScreenPositions))]) #perhaps replace with zip(*sorted((p[2],i) for i,p in enumerate(nodeScreenPositions)))[1] (not sure)
    if raytracingMode:
        raytrace(False,1/2,25,(size[0]/size[1],1),maxReflections=8,antialiasing=0,timeAntialiasing=0,lightSpeed=0,gravityMode=0)
    '''for sc,k,n in zip(nodeScreenPositions,stateTransitions,nodes):
        for l in k:
            drawLine(sc[:2],(average(sc[:2],nodeScreenPositions[l][:2]) if directedMode else nodeScreenPositions[l][:2]),n[3][colourMode])''' #from tablebase vision
    for j,n in zip(nodeScreenPositions,nodes):
        if 2<=n[4]<4:
            drawLine(nodeScreenPositions[0],j,(255,)*3,int(orthographicZoom*n[1][0]))
    if dims==2 or not raytracingMode:
        for i,j,n in [(i,nodeScreenPositions[i],nodes[i]) for i in renderOrder]:
            sezi=2*((j[2] if projectionMode==3 or j[2]==0 else n[1][0]/pixelAngle/j[2]) if dims==3 and perspectiveMode else orthographicZoom*n[1][0]) #different from size
            drawShape(sezi,j[:2],(((0,255,0) if playerLanded else (255,0,0)) if gameMode and i==0 else n[3]),5)
    frames+=1
else: exit()
