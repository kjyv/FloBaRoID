/* Copyright [2016] [Stefan Bethge]
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.*/

#ifndef EXCITATION_MSG
#define EXCITATION_MSG

#include <string>
#include <yarp/os/Portable.h>
#include <yarp/os/Bottle.h>

class excitation_msg
{
public:
    excitation_msg() {
      command = "";
      angle0 = 0.0;
      angle1 = 0.0;
      angle2 = 0.0;
      angle3 = 0.0;
      angle4 = 0.0;
      angle5 = 0.0;
      angle6 = 0.0;
      velocity0 = 0.0;
      velocity1 = 0.0;
      velocity2 = 0.0;
      velocity3 = 0.0;
      velocity4 = 0.0;
      velocity5 = 0.0;
      velocity6 = 0.0;
    }

    std::string command;
    double angle0, angle1, angle2, angle3, angle4, angle5, angle6;
    double velocity0, velocity1, velocity2, velocity3, velocity4, velocity5, velocity6;

    yarp::os::Bottle toBottle() {
        yarp::os::Bottle temp;
        yarp::os::Bottle& list = temp.addList();

        list.addString(command);

        if(command == "set_left_arm")
        {
            list.addDouble(angle0);
            list.addDouble(angle1);
            list.addDouble(angle2);
            list.addDouble(angle3);
            list.addDouble(angle4);
            list.addDouble(angle5);
            list.addDouble(angle6);
            list.addDouble(velocity0);
            list.addDouble(velocity1);
            list.addDouble(velocity2);
            list.addDouble(velocity3);
            list.addDouble(velocity4);
            list.addDouble(velocity5);
            list.addDouble(velocity6);
        }

        return temp;
    }

    void fromBottle(yarp::os::Bottle* temp)
    {
        if (temp->get(0).isNull())
        {
            command="";
            return;
        }
        yarp::os::Bottle* list = temp->get(0).asList();
        if (list==NULL)
        {
            command="";
            return;
        }
        if (list->get(0).isNull())
        {
            command="";
            return;
        }

        //TODO: check that list has enough entries and otherwise give message
        if(list->size() < 7*2) {
            std::cout << "warning: not enough parameters given in Bottle!" << std::endl;
            return;
        }

        command = list->get(0).asString();
        int index = 1;
        if(command == "set_left_arm")
        {
            angle0 = list->get(index++).asDouble();
            angle1 = list->get(index++).asDouble();
            angle2 = list->get(index++).asDouble();
            angle3 = list->get(index++).asDouble();
            angle4 = list->get(index++).asDouble();
            angle5 = list->get(index++).asDouble();
            angle6 = list->get(index++).asDouble();
            velocity0 = list->get(index++).asDouble();
            velocity1 = list->get(index++).asDouble();
            velocity2 = list->get(index++).asDouble();
            velocity3 = list->get(index++).asDouble();
            velocity4 = list->get(index++).asDouble();
            velocity5 = list->get(index++).asDouble();
            velocity6 = list->get(index++).asDouble();
        }

        return;
    }

};

#endif
