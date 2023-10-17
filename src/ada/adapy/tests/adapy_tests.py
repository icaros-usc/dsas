#!/usr/bin/env python
PKG = 'adapy'
import roslib
import numpy, unittest
import adapy, prpy
import prpy.rave


class AdaTest(unittest.TestCase):
  def setUp(self):
    self.env, self.robot = adapy.initialize(sim=True)
    self.robot.arm.SetActive()



  def test_Planning(self):
     values = self.robot.arm.GetDOFValues()
     values[1] = values[1] + 0.1
     self.robot.arm.PlanToConfiguration(values, execute = True)
     

  def test_Hand(self):
     self.robot.arm.hand.OpenHand()
     self.robot.arm.hand.CloseHand()
     self.robot.arm.hand.CloseHandTight()
      

if __name__ == '__main__':
    import rosunit
    rosunit.unitrun(PKG, 'test_ada', AdaTest)
