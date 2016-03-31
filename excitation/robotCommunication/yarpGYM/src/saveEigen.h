/*
 * All rights reserved.
 * Copyright (C) 2014 Przemyslaw Kryczka kryczka.p@gmail.com
 * 
 */



//------------------------------------------------------------------------------------------------
//
//!	@file	saveEigen.h
//!
//!		@date	2012-12-27 created by P. KRYCZKA
//
//------------------------------------------------------------------------------------------------
#ifndef	SAVEEIGEN_H
#define	SAVEEIGEN_H

#include <fstream>
#include <Eigen/Dense>
#include <list>
#include <vector>
#include "saveEigen.h"
#include <string>

// using namespace Eigen;
// using namespace std;

/**
  * @brief  Brief description 
  */

inline void save(Eigen::MatrixXd *mat, std::string filename)
{
	std::ofstream file;
	file.open(filename.c_str(), std::ios::out);
	
	for (int j = 0; j < mat->rows(); j++)
	{	
		for(int i = 0; i < mat->cols()-1 ; i++)  	
			file << (*mat)(j,i) <<  ", " ;	
		file << (*mat)(j,mat->cols()-1)  << "\n";
	}	
	
	file.close();
} 


inline void save_app(Eigen::MatrixXd *mat, std::string filename)
{
	std::ofstream file;
	file.open(filename.c_str(), std::ofstream::out | std::ofstream::app);

	for (int j = 0; j < mat->rows(); j++)
	{	
		for(int i = 0; i < mat->cols()-1 ; i++)  	
			file << (*mat)(j,i) <<  ", " ;	
		file << (*mat)(j,mat->cols()-1)  << "\n";
	}	
	file.close();
}  

inline void save(std::list <Eigen::VectorXd> *listLog, std::string filename)
{
	std::ofstream file;
	file.open(filename.c_str(), std::ofstream::out);

	Eigen::VectorXd tmpVect;
	while (!listLog->empty())
	{	
		tmpVect = listLog->front();
		listLog->pop_front();
		file << tmpVect.transpose() << std::endl;	
		
	}	

	file.close();
}   

inline void save(Eigen::VectorXd *tmpVect, std::string filename)
{
	std::ofstream file;
	file.open(filename.c_str(), std::ofstream::out);
	
	file << *tmpVect << std::endl;	
	
	file.close();
}

inline void save(std::list <Eigen::Vector3d> listLog, std::string filename)
{
	std::ofstream file;
	file.open(filename.c_str(), std::ofstream::out);

	Eigen::Vector3d tmpVect;
	while (!listLog.empty())
	{	
		tmpVect = listLog.front();
		listLog.pop_front();
		file << tmpVect.transpose() << std::endl;	
		
	}	

	file.close();
}   

inline void save(std::list <Eigen::Vector2d> *listLog, std::string filename)
{
	std::ofstream file;
	file.open(filename.c_str(), std::ofstream::out);

	Eigen::Vector2d tmpVect;
	while (!listLog->empty())
	{	
		tmpVect = listLog->front();
		listLog->pop_front();
		file << tmpVect.transpose() << std::endl;	
		
	}	

	file.close();
}   
//--------------------------------------------------------------
//				Non Eigen functions
//--------------------------------------------------------------

inline void save(std::list<std::vector <double> > *listLog, std::string filename)
{
	std::ofstream file;
	file.open(filename.c_str(), std::ofstream::out);

	std::vector<double> tmpVect;
	while (!listLog->empty())
	{	
		tmpVect = listLog->front();
		listLog->pop_front();
		for (int i=0; i<tmpVect.size()-1; i++) {
		file << tmpVect[i] << ",";
		}
		file << tmpVect[tmpVect.size()-1] << std::endl;
	}	

	file.close();
} 

inline void save(std::list<double> *listLog, std::string filename)
{
	std::ofstream file;
	file.open(filename.c_str(), std::ofstream::out);

	double tmpValue;
	while (!listLog->empty())
	{	
		tmpValue = listLog->front();
		listLog->pop_front();
		file << tmpValue << std::endl;
	}	

	file.close();
} 

inline void save(std::list<int> *listLog, std::string filename)
{
	std::ofstream file;
	file.open(filename.c_str(), std::ofstream::out);

	int tmpValue;
	while (!listLog->empty())
	{	
		tmpValue = listLog->front();
		listLog->pop_front();
		file << tmpValue << std::endl;
	}	

	file.close();
} 


inline void save(std::vector<double> *listLog, std::string filename)
{
	std::ofstream file;
	file.open(filename.c_str(), std::ofstream::out);

	double tmpValue;
	for (int i=0; i<listLog->size();i++)
	{	
		file << listLog->at(i) << std::endl;
	}	

	file.close();
} 

inline void save(std::vector<int> *listLog, std::string filename)
{
	std::ofstream file;
	file.open(filename.c_str(), std::ofstream::out);

	double tmpValue;
	for (int i=0; i<listLog->size();i++)
	{	
		file << listLog->at(i) << std::endl;
	}	

	file.close();
} 

inline void save(int number, std::string filename)
{
	std::ofstream file;
	file.open(filename.c_str(), std::ofstream::out);
	
	file << number;

	file.close();
} 


#endif