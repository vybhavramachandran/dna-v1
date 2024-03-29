import os
import subprocess
import time
import signal
import sys
import math
import numpy as np
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding, EzPickle
#import Box2D.examples.simple.rendering as rendering
import Box2D  # The main library
from Box2D.b2 import (world, polygonShape, staticBody,
                      dynamicBody, color, draw, contactListener, fixtureDef)
import re


class ContactDetector(contactListener):

    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def isItHittingTheWall(self, body1, body2):
        if "Nucleus" in body1.userData or "Nucleus" in body2.userData:
            return True
        else:
            return False

    def BeginContact(self, contact):
        # if contact is between RPOL and Nucleotide
        if self.env.startContact == False:
            self.env.startContact = True
            print("Begin Contact between {0} and {1}".format(
                contact.fixtureA.body.userData, contact.fixtureB.body.userData))
            if self.isItHittingTheWall(contact.fixtureA.body, contact.fixtureB.body) == True:
                self.env.game_over = True
            else:
                if contact.fixtureA.body.userData == "RPOL2":
                    self.env.mRNA += contact.fixtureB.body.userData

                else:
                    self.env.mRNA += contact.fixtureA.body.userData

    def EndContact(self, contact):
        # if RPOL breaks contact with one Nucleotide
        self.env.startContact = False
        print("End Contact between {0} and {1}".format(
            contact.fixtureA.body.userData, contact.fixtureB.body.userData))


class DnaEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        EzPickle.__init__(self)
        self.PPM = 20.0  # pixels per meter
        self.TARGET_FPS = 60
        self.FPS = 60
        self.SCALE = 4.0

        self.TIME_STEP = 1.0 / self.TARGET_FPS
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 480
        self.FIRST_NUCLEOTIDE_X = 15
        self.FIRST_NUCLEOTIDE_Y = 40

        self.VIEWPORT_W = 600
        self.VIEWPORT_H = 400

        self.viewer = None

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)

        self.atpConsumed = 0

        self.world = world(gravity=(0, 0), doSleep=True)

        self.prev_reward = None

        self.templateDNAStrand = "CAG"
        self.mRNA = ""

        # useful range is -1 .. +1, but spikes can be higher
        # self.observation_space = spaces.Box(-np.inf,
        #                                    np.inf, shape=(8,), dtype=np.float32)
        # actions are
        # 0 : NoP
        # 1 : Move up
        # 2 : Move down
        # 3 : Move left
        # 4 : Move right
        # 5 : Transcribe
        self.action_space = spaces.Discrete(6)
        self.colors = {
            "ground": (0, 0, 0),
            "A": (127, 0, 127),
            "G": (0, 255, 0),
            "T": (127, 0, 127),
            "C": (127, 0, 127),
            "RPOL2": (127, 0, 127)
        }

        self.reset()

    def _destroy(self):
        self.world.contactListener = None
        for body in self.world.bodies:
            self.world.DestroyBody(body)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def createNucleus(self):
        # self.nucleusBody = self.world.CreateStaticBody(
        #     position=(self.FIRST_NUCLEOTIDE_X - 5 , self.FIRST_NUCLEOTIDE_Y+20), angle=0, userData="Nucleus")
        # self.nucleusShape = b2PolygonShape(
        #     vertices=[(0, 0), (60, 0), (60, -40), (0, -40)]
        # )
        # self.nucleusFixture = b2FixtureDef(
        #     shape=self.nucleusShape, density=1, friction=0.3)
        # box = self.nucleusBody.CreateFixture(self.nucleusFixture)
        # self.nucleusPolyLineCoordinates = []
        # self.nucleusPolyLineCoordinates = [
        #     b2Vec2(100, 18), b2Vec2(100, 50), b2Vec2(10, 50), b2Vec2(10, 18)]
        # print(self.nucleusPolyLineCoordinates)

        self.nucleusTopBorderBody = self.world.CreateStaticBody(
            position=(self.FIRST_NUCLEOTIDE_X - 5, self.FIRST_NUCLEOTIDE_Y+20), angle=0, userData="NucleusTopBorder")
        self.nucleusRightBorderBody = self.world.CreateStaticBody(
            position=(self.FIRST_NUCLEOTIDE_X + 55, self.FIRST_NUCLEOTIDE_Y+20), angle=0, userData="NucleusRightBorder")
        self.nucleusLeftBorderBody = self.world.CreateStaticBody(
            position=(self.FIRST_NUCLEOTIDE_X - 5, self.FIRST_NUCLEOTIDE_Y+20), angle=0, userData="NucleusLeftBorder")
        self.nucleusBottomBorderBody = self.world.CreateStaticBody(
            position=(self.FIRST_NUCLEOTIDE_X - 5, self.FIRST_NUCLEOTIDE_Y-20), angle=0, userData="NucleusBottomBorder")

        self.nucleusTopBorderShape = polygonShape(
            vertices=[(0, 0), (60, 0), (60, 1), (0, 1)]
        )
        self.nucleusRightBorderShape = polygonShape(
            vertices=[(0, 0), (1, 0), (1, -40), (0, -40)]
        )
        self.nucleusLeftBorderShape = polygonShape(
            vertices=[(0, 0), (1, 0), (1, -40), (0, -40)]
        )
        self.nucleusBottomBorderShape = polygonShape(
            vertices=[(0, 0), (60, 0), (60, 1), (0, 1)]
        )

        self.nucleusTopBorderFixture = fixtureDef(
            shape=self.nucleusTopBorderShape, density=1, friction=0.3)
        self.nucleusLeftBorderFixture = fixtureDef(
            shape=self.nucleusLeftBorderShape, density=1, friction=0.3)
        self.nucleusRightBorderFixture = fixtureDef(
            shape=self.nucleusRightBorderShape, density=1, friction=0.3)
        self.nucleusBottomBorderFixture = fixtureDef(
            shape=self.nucleusBottomBorderShape, density=1, friction=0.3)

        box = self.nucleusTopBorderBody.CreateFixture(
            self.nucleusTopBorderFixture)
        box = self.nucleusRightBorderBody.CreateFixture(
            self.nucleusRightBorderFixture)
        box = self.nucleusLeftBorderBody.CreateFixture(
            self.nucleusLeftBorderFixture)
        box = self.nucleusBottomBorderBody.CreateFixture(
            self.nucleusBottomBorderFixture)

        return self.nucleusTopBorderBody, self.nucleusLeftBorderBody, self.nucleusRightBorderBody, self.nucleusBottomBorderBody

    def createRNANucleotide(self):
        print("Placeholder")

    def createRNAPhosphateJoint(self, base1, base2):
        print("Placeholder")

    def createPhosphateBond(self, base1, base2):
        print(base1.userData, base2.userData)
        self.world.CreateDistanceJoint(bodyA=base1, bodyB=base2, anchorA=base1.worldCenter,
                                       anchorB=base2.worldCenter, collideConnected=True)
        # basePairMainShape = b2PolygonShape(vertices=[
        #     (-1, 2), (-1, -2), (1, -2), (1, 2)
        # ])

    def createBackbone(self, list_of_bases):
        for i in range(len(list_of_bases)-1):
            self.createPhosphateBond(list_of_bases[i], list_of_bases[i+1])

    def createRNAPolymerase(self):
        self.rnaPolymeraseBody = self.world.CreateDynamicBody(
            position=(50, 50), angle=0, userData="RPOL2", fixedRotation=True)
        self.rnaPolymeraseShape = polygonShape(vertices=[
            (-1, 2), (-1, -2), (1, -2), (1, 2)
        ])
        self.rnaPolymeraseFixture = fixtureDef(
            shape=self.rnaPolymeraseShape, density=1, friction=0.3)
        box = self.rnaPolymeraseBody.CreateFixture(self.rnaPolymeraseFixture)
        return self.rnaPolymeraseBody

    def createNucleotideOfType(self, type, x_offset, y_offset):
        self.createdBody = self.world.CreateStaticBody(
            position=(self.FIRST_NUCLEOTIDE_X + x_offset, self.FIRST_NUCLEOTIDE_Y+y_offset), angle=0, userData=type)

        self.basePairMainShape = polygonShape(vertices=[
            (-1, 2), (-1, -2), (1, -2), (1, 2)
        ])

        self.basePairLeftShape = polygonShape(vertices=[
            (-1, -1.5), (-1.5, -1.5), (-1.5, -2), (-1, -2)
        ])

        self.basePairRightShape = polygonShape(vertices=[
            (1, -1.5), (1.5, -1.5), (1.5, -2), (1, -2)
        ])

        # And add a box fixture onto it (with a nonzero density, so it will move)
        self.basePairMainFixture = fixtureDef(
            shape=self.basePairMainShape, density=1, friction=0.3)

        self.basePairLeftFixture = fixtureDef(
            shape=self.basePairLeftShape, density=1, friction=0.3)

        self.basePairRightFixture = fixtureDef(
            shape=self.basePairRightShape, density=1, friction=0.3)

        box = self.createdBody.CreateFixture(self.basePairMainFixture)
        box = self.createdBody.CreateFixture(self.basePairLeftFixture)
        box = self.createdBody.CreateFixture(self.basePairRightFixture)
        # box2 = dynamic_body2.CreatePolygonFixture(box=(1,2), density=1, friction=0.3)
        return self.createdBody

    def drawTemplateStrand(self, bases):
        # bases = "CCAAACA"
        self.renderedBases = []
        self.x_offset_to_add = 4
        self.y_offset_to_add = 0
        for i in range(len(bases)):
            self.renderedBases.append(self.createNucleotideOfType(
                bases[i], self.x_offset_to_add*i, self.y_offset_to_add))
        return self.renderedBases

    def step(self, action):
        # test
        # print("step called", action)
        m_power = 1.0
        print("mRNA is ", self.mRNA)
        assert self.action_space.contains(
            action), "%r (%s) invalid " % (action, type(action))
        # handle action : Move UP (1)
        if action == 1:
            self.rnaPolymerase.ApplyLinearImpulse(
                (0, 50), self.rnaPolymerase.worldCenter, True)

        # handle action : Move DOWN (2)
        if action == 2:
            self.rnaPolymerase.ApplyLinearImpulse(
                (0, -50), self.rnaPolymerase.worldCenter, True)

        # handle action : Move Left (3)
        if action == 3:
            self.rnaPolymerase.ApplyLinearImpulse(
                (-50, 0), self.rnaPolymerase.worldCenter, True)

        # handle action : Move Right (4)
        if action == 4:
            self.rnaPolymerase.ApplyLinearImpulse(
                (50, 0), self.rnaPolymerase.worldCenter, True)

        # handle action : Transcribe (5)
        if action == 5:
            print("Placeholder for action 5")

        self.world.Step(1.0/self.FPS, 6*30, 2*30)
        pos = self.rnaPolymerase.position
        # print(type(pos.x))
        # vel = self.rnaPolymerase.linearVelocity
        state = [pos.x, pos.y, self.renderedBases[0].position.x,
                 self.renderedBases[0].position.y]
        print("Length of state vector",len(state))
        if len(self.mRNA) <= len(self.templateDNAStrand):
            if re.findall("^"+self.mRNA, self.templateDNAStrand) != []:
                if re.findall("^"+self.mRNA+"$", self.templateDNAStrand) != []:
                    print("Finished")
                    done = True
                    self.reward = 1000
                else:
                    print(self.mRNA, self.templateDNAStrand)
                    self.reward = 10
                    done = False
            else:
                done = True
                self.game_over = True

        else:
            done = True
            self.game_over = True

        if self.game_over:
            done = True
            self.reward = -100
       
        return np.array(state, dtype=np.float32), self.reward, done, {}

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.startContact = False
        print("reset called new")
        self.rnaPolymerase = self.createRNAPolymerase()
        self.nucleus = self.createNucleus()
        # self.createNucleus()
        self.game_over = False
        self.mRNA = ""
        self.reward = 0
        self.templateDNAStrand = "CAG"

        self.atpConsumed = 0
        # Create the list of nucleotides
        self.nucleotidesToRender = self.drawTemplateStrand(
            self.templateDNAStrand)

        # Create Phosphate backbone
        self.createBackbone(self.nucleotidesToRender)

        self.nucleotidesToRender.append(self.rnaPolymerase)
        # print("Number of nucleotides to print are",len(self.nucleotidesToRender))
        return self.step(0)[0]

    def render(self, mode='human', close=False):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.VIEWPORT_W, self.VIEWPORT_H)
            self.viewer.set_bounds(
                0, self.VIEWPORT_W/self.SCALE, 0, self.VIEWPORT_H/self.SCALE)

         # Draw the nucleus
        for body in self.nucleus:
            for fixture in body.fixtures:
                trans = fixture.body.transform
                path = [trans*v for v in fixture.shape.vertices]
                path.append(path[0])
               # print("Vertices", path)
                self.viewer.draw_polygon(path, color=(0.8, 0.8, 0.8))

        # self.viewer.draw_polyline(
        #     self.nucleusPolyLineCoordinates, color=(127,0,127))

        for body in self.nucleotidesToRender:
            # viewer.draw_polygon(body,color=(0,0,0))  # or: world.bodies
            # The body gives us the position and angle of its shapes
            for fixture in body.fixtures:
                nucleotideColor = self.colors[body.userData]
                trans = fixture.body.transform
                path = [trans*v for v in fixture.shape.vertices]
                # print(path)
                self.viewer.draw_polygon(path, color=nucleotideColor)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
