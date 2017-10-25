/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
/*                                                                                                              */
/*   B L A N K   A P P L I C A T I O N                                                                          */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// System includes
#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <algorithm>
#include <stdint.h>
#include <limits.h>
#include <stdexcept>

// OpenNN includes

#include "opennn/opennn.h"
#include <opennn/multilayer_perceptron.h>
#include <opennn/perceptron_layer.h>
#include "training_strategy.h"
//#include "tests/*.h"
#include "variables.h"
//#include "statistics.h"
using namespace OpenNN;
using namespace std;

int main(void)
{

	try
	{
		std::cout << "OpenNN. Blank Application." << std::endl;

		srand((unsigned int)time(NULL));

		/*DATA SET*/
		DataSet data_set;

		data_set.set_data_file_name("data/dummyDrumData.dat");

		data_set.set_separator("Comma");

		cout << "about to load data" << endl;

		data_set.load_data();

		cout << "loaded data" << endl;

		/*VARIABLES */

		Variables* variables_pointer = data_set.get_variables_pointer();

		variables_pointer->set_name(0, "MFCC 0");
		variables_pointer->set_use(0, Variables::Input);

		variables_pointer->set_name(1, "MFCC 3");
		variables_pointer->set_use(1, Variables::Input);

		variables_pointer->set_name(2, "MFCC 6");
		variables_pointer->set_use(2, Variables::Input);

		variables_pointer->set_name(3, "MFCC 7");
		variables_pointer->set_use(3, Variables::Input);

		variables_pointer->set_name(4, "MFCC 9");
		variables_pointer->set_use(4, Variables::Input);

		variables_pointer->set_name(5, "Peak Ratio Chroma E");
		variables_pointer->set_use(5, Variables::Input);

		variables_pointer->set_name(6, "Peak Ratio Chroma F");
		variables_pointer->set_use(6, Variables::Input);

		variables_pointer->set_name(7, "Peak Ratio Chroma F Sharp");
		variables_pointer->set_use(7, Variables::Input);

		variables_pointer->set_name(8, "Centroid Power");
		variables_pointer->set_use(8, Variables::Input);

		variables_pointer->set_name(9, "MFCC 0 NUMBER 2"); //WTF
		variables_pointer->set_use(9, Variables::Input);

		variables_pointer->set_name(10, "MFCC 8"); 
		variables_pointer->set_use(10, Variables::Input);

		variables_pointer->set_name(11, "MFCC 9 NUMBER 2"); //WTF
		variables_pointer->set_use(11, Variables::Input);

		variables_pointer->set_name(12, "Peak Ratio Chroma D"); 
		variables_pointer->set_use(12, Variables::Input);

		variables_pointer->set_name(13, "Peak Ratio Minimum Chroma A"); 
		variables_pointer->set_use(13, Variables::Input);

		variables_pointer->set_name(14, "Zero Crossings"); 
		variables_pointer->set_use(14, Variables::Input);

		variables_pointer->set_name(15, "MFCC 7 NUMBER 2"); //WTF
		variables_pointer->set_use(15, Variables::Input);

		variables_pointer->set_name(16, "MFCC 10");
		variables_pointer->set_use(16, Variables::Input);

		variables_pointer->set_name(17, "Peak Ratio Chroma B"); 
		variables_pointer->set_use(17, Variables::Input);

		variables_pointer->set_name(18, "Peak Ratio Chroma C");
		variables_pointer->set_use(18, Variables::Input);

		variables_pointer->set_name(19, "Peak Ratio Chroma F NUMBER 2"); // WTF
		variables_pointer->set_use(19, Variables::Input);

		variables_pointer->set_name(20, "Center");
		variables_pointer->set_use(20, Variables::Target);

		variables_pointer->set_name(21, "Halfedge");
		variables_pointer->set_use(21, Variables::Target);

		variables_pointer->set_name(22, "Rimshot");
		variables_pointer->set_use(22, Variables::Target);

		const Matrix<std::string> inputs_information = variables_pointer->arrange_inputs_information();
		const Matrix<std::string> targets_information = variables_pointer->arrange_targets_information();

		/*SCALING THE INPUT DATA FOR BETTER PERFORMANCE*/
		Instances* instances_pointer = data_set.get_instances_pointer();

		instances_pointer->split_random_indices(); //must select some data to use for trianing
												   //and some for testing
												   //openNN/ example does random split as well.

		const Vector< Statistics<double> > inputs_statistics = data_set.scale_inputs_minimum_maximum();

		/*SCALING THE INPUT DATA FOR APPROXIMATION PROBLEMS*/
		const Vector< Statistics<double> > targets_statistics = data_set.scale_targets_minimum_maximum();
		/*MIGHT NOT BE NECESSARY IN OUR CASE... ???*/

		/*OBTAIN AND SET INFO ABOUT THE VARIABLES OF THE DATA*/
		//i guess we still need to do this??
		//Variables* variables_pointer = data_set.get_variables_pointer();
		const size_t inputs_number = variables_pointer->count_inputs_number();//should give 20
		cout << "\n inputs number : " << inputs_number << endl;
		const size_t outputs_number = variables_pointer->count_targets_number(); //should give 3
		cout << "\n outputs number: " << outputs_number << endl;
		/*----------------------------------*/
		/*CREATING A NN WITH ONLY ONE HIDDEN LAYER.*/
		/*CREATE A SCALINGN LAYER FOR IT BASED ON THE DATA SET STATISTICS*/
		/*Should be 20, something, 3 */
		NeuralNetwork neural_network(inputs_number, 30, outputs_number);

		neural_network.construct_scaling_layer();

		ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

		scaling_layer_pointer->set_statistics(inputs_statistics); /*WTF IS THIS?*/

		scaling_layer_pointer->set_scaling_method(ScalingLayer::NoScaling);

		/*
		FOR REGRESSION PROBLEMS
		neural_network.construct_unscaling_layer();

		UnscalingLayer* unscaling_layer_pointer = neural_network.get_unscaling_layer_pointer();

		unscaling_layer_pointer->set_statistics(targets_statistics);

		unscaling_layer_pointer->set_unscaling_method(UnscalingLayer::NoUnscaling);*/


		/*PROBABILISTIC LAYER - FOR PATTERN RECOGNITION & CLASSIFICATION NN'S*/
		neural_network.construct_probabilistic_layer();

		ProbabilisticLayer* probability_layer_pointer = neural_network.get_probabilistic_layer_pointer();

		probability_layer_pointer->set_probabilistic_method(ProbabilisticLayer::Probability);

		//in the example they use Softmax instead of Probability

		/*ERROR*/
		LossIndex loss_index(&neural_network, &data_set);//set both pointers at same time!
		loss_index.set_error_type("NORMALIZED_SQUARED_ERROR"); //not in the example.
		loss_index.set_regularization_type(LossIndex::NEURAL_PARAMETERS_NORM);

		/*TRAINING STRATEGY*/
		TrainingStrategy training_strategy(&loss_index);
		training_strategy.set_main_type(TrainingStrategy::QUASI_NEWTON_METHOD);
		QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();
		quasi_Newton_method_pointer->set_maximum_iterations_number(30); //changed this.
		quasi_Newton_method_pointer->set_display_period(10);
		training_strategy.perform_training();
		/*TESTING ANALYSYS - linear regression*/
		/*Not needed in our case, NOT regressing!!!*/
		/*TestingAnalysis testing_analysis(&neural_network, &data_set);
		TestingAnalysis::LinearRegressionResults linear_regression_results = testing_analysis.perform_linear_regression_analysis();*/

		/*TESTING ANALYSYS - Confusion Matrix*/
		TestingAnalysis testing_analysis(&neural_network, &data_set);
		Matrix<size_t> confusion = testing_analysis.calculate_confusion();

		scaling_layer_pointer->set_scaling_method(ScalingLayer::MinimumMaximum);

		/*
		For regression problems !
		unscaling_layer_pointer->set_unscaling_method(UnscalingLayer::MinimumMaximum);
		*/

		neural_network.save_expression("expresion.txt");

		//neural_network.save_expression_python("expresion.py");
		//neural_network.save_expression_R("expresion.R");

		/*from openNN/ examples */
		/*the example had a model selection -- we didn't*/
		data_set.save("data/proceeds/data_set.xml");

		neural_network.save("data/proceeds/neural_network.xml");
		neural_network.save_expression("data/proceeds/expression.txt");
		confusion.save("data/proceeds/confusion.dat");

		training_strategy.save("data/proceeds/training_strategy.xml");

		//model_selection.save("diff_data/model_selection.xml");
		// COMMENTED IN EXAMPLE TOO model_selection_results.save("../data/model_selection_results.dat");


		return(0);
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;

		return(1);
	}

}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2016 Roberto Lopez.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
