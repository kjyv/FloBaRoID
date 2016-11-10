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
      //right
      angle0 = 0.0;
      angle1 = 0.0;
      angle2 = 0.0;
      angle3 = 0.0;
      angle4 = 0.0;
      angle5 = 0.0;
      //left
      angle6 = 0.0;
      angle7 = 0.0;
      angle8 = 0.0;
      angle9 = 0.0;
      angle10 = 0.0;
      angle11 = 0.0;
    }

    std::string command;
    double angle0, angle1, angle2, angle3, angle4, angle5;
    double angle6, angle7, angle8, angle9, angle10, angle11;

    yarp::os::Bottle toBottle() {
        yarp::os::Bottle temp;
        yarp::os::Bottle& list = temp.addList();

        list.addString(command);

        if(command == "set_legs_refs")
        {
            list.addDouble(angle0);
            list.addDouble(angle1);
            list.addDouble(angle2);
            list.addDouble(angle3);
            list.addDouble(angle4);
            list.addDouble(angle5);
            list.addDouble(angle6);
            list.addDouble(angle7);
            list.addDouble(angle8);
            list.addDouble(angle9);
            list.addDouble(angle10);
            list.addDouble(angle11);
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

        //TODO: check that list has enough entries and otherwise give error message
        if(list->size() < 6*2) {
            std::cout << "warning: not enough parameters given in Bottle!" << std::endl;
            return;
        }

        command = list->get(0).asString();
        int index = 1;
        if(command == "set_legs_refs")
        {
            angle0 = list->get(index++).asDouble();
            angle1 = list->get(index++).asDouble();
            angle2 = list->get(index++).asDouble();
            angle3 = list->get(index++).asDouble();
            angle4 = list->get(index++).asDouble();
            angle5 = list->get(index++).asDouble();
            angle6 = list->get(index++).asDouble();
            angle7 = list->get(index++).asDouble();
            angle8 = list->get(index++).asDouble();
            angle9 = list->get(index++).asDouble();
            angle10 = list->get(index++).asDouble();
            angle11 = list->get(index++).asDouble();
        }

        return;
    }

};

#endif
