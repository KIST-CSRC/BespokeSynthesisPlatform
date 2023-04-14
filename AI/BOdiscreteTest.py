#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ##
# @brief    [BOdisceteTest.py] Bayesian Optimization Discrete file
# @Inspiration
    # https://github.com/fmfn/BayesianOptimization
    # https://github.com/CooperComputationalCaucus/kuka_optimizer
# @author   Hyuk Jun Yoo (yoohj9475@kist.re.kr)   
# @version  1_2   
# TEST 2021-11-01
# TEST 2022-04-11

from bdb import GENERATOR_AND_COROUTINE_FLAGS
from cmath import nan
from email import iterators
from types import new_class
from skopt.sampler import Grid
from skopt.sampler import Lhs
from skopt.space import Space
from scipy.spatial.distance import pdist
import random
import json
# from random import random
from re import X
import numpy as np
import time
from collections import Counter
import pickle
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from AI.Bayesian.bayesian_optimization import DiscreteBayesianOptimization,Events
from AI.Bayesian.target_space import DiscreteSpace, PropertySpace
from AI.Bayesian.util import UtilityFunction
from AI.Earlystopping import EarlyStopping_AutoLab


class ASLdiscreteBayesianOptimization(DiscreteBayesianOptimization):
    """
    This class Discrete Bayesian Optimization for generating synthesis condition in autonomous laboratory.

    :param algorithm_dict (dict) : include all of information about algorithm config
    {
        "sampling_method" : "grid"
        "initRandom":1,
        "batchSize":8,
        "algorithmType":"BayesianOptimization",
        "verbose":0,
        "sampler":"greedy",
        "utilityType":["ucb"],
        "randomState":2,
        "targetConditionDict":
            {
                "GetUVdata":
                    {
                        "Property":{"lambdamax":550,"FWHM":100},
                        "Ratio":{"lambdamax":1,"FWHM":0}
                    }
            },
        "prangeDict" : 
        {
            "AgNO3" : [500, 3000, 100],
            "NaBH4": [500, 3000, 100]
        }
    }
    :param kwargs: --> suggestion = disc_constrained_acq_max(
                    ac=utility_function.utility,
                    instance=self,
                    **kwargs)
    Sub_Params
        :param prangeDict (dict): dict of 3-tuples for variable min, max, step
        :param verbose=2 (int): verbosity
        :param batchSize=8 (int): number of points in a batch (1 cycle에 몇개의 batch 합성을 할 것인가?)
        :param initRandom=1 (int): number of random batches (cycle 초반에 얼마나 random하게 진행할 것인가?)
        :param randomState=1 (int): random seed
        :param sampler=greedy (str): "greedy" or "KMBBO" or "capitalist"
        :param targetConditionDict (dict) : calculate loss
        :param sampling_method (str) : "grid" or "latin" or " random" (cycle 초반의 sampling을 어떤 방법으로 진행할 것인가?)
        :param initParameterList=[] (list) : initial condition
        :param constraints=[] (list) : constraint condition
        :param expectedMax=None: float, expected maximum for proximity return (얼마나 max인 것을 찾을 것이냐? )
    """
    def __init__(self, algorithm_dict, **kwargs):
        self.prange=algorithm_dict["prangeDict"]
        self.verbose=algorithm_dict["verbose"]
        self.batch_size=algorithm_dict["batchSize"]
        self.sampling_method = algorithm_dict["sampling_method"]
        self.init_random=algorithm_dict["initRandom"]
        self.sampler=algorithm_dict["acquisitionFunc"]["sampler"]
        self.random_state=algorithm_dict["randomState"]
        self.verbose = algorithm_dict["verbose"]
        self.utility_type=algorithm_dict["acquisitionFunc"]["utilityType"]
        self.acquistionFunc_hyperparameter=algorithm_dict["acquisitionFunc"]["hyperparameter"]
        self.constraints=[]
        self.init_parameter_list=None
        self.expected_max=None
        self.targetConditionDict=algorithm_dict["targetConditionDict"]

        if "initParameterList" in algorithm_dict:
            self.init_parameter_list=algorithm_dict["initParameterList"]
        if "constraints" in algorithm_dict:
            self.constraints=algorithm_dict["constraints"]
        if "expectedMax" in algorithm_dict:
            self.expectedMax=algorithm_dict["expectedMax"]
        
        if self.verbose:
            self._prime_subscriptions()
            self.dispatch(Events.OPTMIZATION_START)

        self.normPrange=self._getNormalizeList(self.prange)
        DiscreteBayesianOptimization.__init__(self, f=None,
            prange=self.normPrange,
            verbose=int(self.verbose),
            random_state=self.random_state,
            constraints=self.constraints)

        self._real_space = DiscreteSpace(target_func=None, prange=self.prange, random_state=self.random_state)
        self._property_space = PropertySpace(pbounds=self.prange, target_condition_dict=self.targetConditionDict)

        self.ucb_utility = UtilityFunction(kind='ucb', kappa=self.acquistionFunc_hyperparameter)
        self.ei_utility = UtilityFunction(kind='ei', xi=self.acquistionFunc_hyperparameter)
        self.poi_utility = UtilityFunction(kind='poi')
        self.es_utility = UtilityFunction(kind='es')
        self.max_val = {'proximity': -1, 'iter': 0, 'val': 0}
        self.process_start_time = time.time()

        self.earlystopping=EarlyStopping_AutoLab()

    def _getNormalizeList(self, prangeDict):
        '''
        :param prangeDict (dict) : 
        
        ex)
            {
                "AgNO3" : [100, 3000, 50],
                "H2O2" : [100, 3000, 50],
                "NaBH4": [100, 3000, 50]
            }

        :return : normPrangeDict
        '''
        normPrangeDict={}
        for chemical, rangeList in prangeDict.items():
            new_range_list=[]
            
            new_range_list.append(0) # normalize min value = 0
            new_range_list.append(1) # normalize max value = 1
            new_range_list.append(rangeList[2]/(rangeList[1]-rangeList[0]))

            normPrangeDict[chemical] = new_range_list
        return normPrangeDict

    def _getNormalizedCondition(self, real_next_points):
        """
        convert real condition to normalized condition

        :param real_next_points (list) : 
            [
                {'AgNO3': 3300.0, 'Citrate': 500.0, 'H2O': 1300.0, 'H2O2': 3500.0, 'NaBH4': 3100.0}
                {'AgNO3': 3500.0, 'Citrate': 500.0, 'H2O': 1300.0, 'H2O2': 3500.0, 'NaBH4': 3500.0}
                {'AgNO3': 800.0,  'Citrate': 500.0, 'H2O': 1300.0, 'H2O2': 3500.0, 'NaBH4': 3500.0}
                {'AgNO3': 3500.0, 'Citrate': 500.0, 'H2O': 3500.0, 'H2O2': 3500.0, 'NaBH4': 3500.0}
            ]

        :return : normalized_next_points (list)
        """
        normalized_next_points = []
        for _, next_point in enumerate(real_next_points):
            new_value={}
            for chemical, rangeList in self.prange.items():
                new_value[chemical]=(next_point[chemical]-rangeList[0])/(rangeList[1]-rangeList[0]) # X' = (value - V_min)/(V_max - V_min) 
            normalized_next_points.append(new_value)
        
        return normalized_next_points

    def _getRealCondition(self, normalized_next_points):
        """
        convert normalized condition to real condition

        :param normalized_next_points (list) : 
            [
                {'AgNO3': 0.02123154896, 'Citrate': 0.4563211120887, 'H2O': 0.122471125, 'H2O2': 0.6337412354, 'NaBH4': 1.0}
                ...
            ]

        :return : real_next_points (list)
        """
        real_next_points = []
        for _, normalized_next_point in enumerate(normalized_next_points):
            new_value={}
            for chemical, rangeList in self.prange.items():
                new_value[chemical]=round(normalized_next_point[chemical]*(rangeList[1]-rangeList[0])+rangeList[0])
            real_next_points.append(new_value)
            
        return real_next_points

    def _earlystopping(self, fitness_list):
        """
        :param iter_num (int): iter_num
        :param fitness (int): fitness
        :param bound_expected_max=0.97 (int) : result > bound_expected_max * self.expected_max

        :return : None
        """
        for fitness in fitness_list:
            earlystoppping_type = self.earlystopping(fitness)

        return earlystoppping_type
    
    def _checkCandidateNumber(self, count, candidate_list):
        """
        to check and cut candidate in "candidate_list" depending on "count" nunmber

        :param count (int): the number of candidate
        :param candidate_list (list): candidate_list. ex) [{}, {}, {}, {} ... ]

        :return candidate_list, trash_list (list, list): candidate_list, trash_list
        """
        trash_list = []
        if len(candidate_list) != count:
            for _ in range(len(candidate_list)-count):
                trash = candidate_list.pop()
                trash_list.append(trash)

        return candidate_list, trash_list

    def _getSamplingList(self,sampling_method):
        """
        :param sampling_method (str) : grid or random or latin

        :return sampling_list (list) : return grid, random or latin sampling list
        """
        if sampling_method == "grid":
            if len(self.space.bounds) == 1:
                n_samples = self.batch_size*(self.init_random +1 )
                targetrng = self.space.bounds.tolist()[0]
                sample = []
                sampling_list = []
                if n_samples > 1:
                    for  i in range(n_samples):
                        step = (max(targetrng) - min(targetrng))/(n_samples-1)
                        sampling = [min(targetrng) + step* i ]
                        sample.append(sampling)
                    for i in range(len(sample)):
                        round_sample= self.space._bin(sample[i])
                        sampling_list.append(self._space.array_to_params(round_sample))
                elif n_samples == 1:
                    sample = random.choice([np.array([500]), np.array([3000])]) 
                    round_sample= self.space._bin(sample)
                    
                    sampling_list.append(self._space.array_to_params(round_sample))
            else :                
                sample =self._space.grid_sample(n_samples= (self.init_random + 1) * self.batch_size).tolist()
                sampling_list = [] 
                for i in range(len(sample)):
                    round_sample= self.space._bin(sample[i])
                    sampling_list.append(self._space.array_to_params(round_sample))

        elif sampling_method == "latin":
            sample =self._space.latin_sample(n_samples= (self.init_random +1) * self.batch_size).tolist()        
            sampling_list = [] 
            for i in range(len(sample)):
                
                round_sample= self.space._bin(sample[i])
                
                sampling_list.append(self._space.array_to_params(round_sample))
        elif sampling_method == "random":          
            sampling_list = [self._space.array_to_params(self.space._bin(
                        self._space.random_sample(constraints=self.get_constraint_dict()))) for _ in range(self.batch_size)]

        return sampling_list
    
    def _register(self, space, params, target):
        """
        :param space (obj) : object of TargetSpace class
        :param params (list) : params_list
        :param target (list) : targets_list

        :return None :
        """
        space.register(params, target)

    def _extractParamsTarget(self, space_res_list):
        """
        extract params and target in space
        :param space_res_list (list) : space_res_list

        :return params_list, target_list (list, list): params_list, target_list
        """
        params_list=[]
        target_list=[]
        for res in space_res_list:
            params_list.append(res["params"])
            target_list.append(res["target"])
        
        return params_list, target_list

    def suggestNextStep(self, utility_type ,iter_num):
        """
        :param utility_type (str or list): -> "ei", "poi", "es", "ucb" or ['ucb', 'ucb', 'ucb', 'ucb', 'ei', 'ei', 'es', 'es']
        :param iter_num (int): iteration number. if n_batches=10, iter_num (1~10)

        :return next_points (dict): candidate of condition
        """
        if iter_num < self.init_random :  # initial sampling 
            if self.init_parameter_list == None: # no initial synthesis condition
                norm_next_points = [self._space.array_to_params(self.space._bin(
                    self._space.random_sample(constraints=self.get_constraint_dict()))) for _ in range(self.batch_size)]
            else: # yes initial synthesis condition
                if iter_num == 0: # iter_num= 0이면 config file에 있는 조건 실행
                    norm_next_points=self.init_parameter_list
                else:
                    norm_next_points = [self._space.array_to_params(self.space._bin(
                        self._space.random_sample(constraints=self.get_constraint_dict()))) for _ in range(self.batch_size)]
        else: # Bayesian optimization with earlystopping
            if type(utility_type) == str:
                if utility_type == 'ucb':
                    temp_utility = self.ucb_utility
                elif utility_type == 'ei':
                    temp_utility = self.ei_utility
                elif utility_type == 'poi':
                    temp_utility = self.poi_utility
                elif utility_type == 'es':
                    temp_utility = self.es_utility
                norm_next_points = self.suggest(temp_utility,sampler=self.sampler,
                                        n_acqs=self.batch_size) # return list
                if len(norm_next_points) > self.batch_size:
                    norm_next_points, trash_list = self._checkCandidateNumber(self.batch_size, norm_next_points)
                elif len(norm_next_points) < self.batch_size:
                    raise ValueError("Candiate of condition is not match to batch_size. norm_next_points: {} != batch_size: {}. Please check this part.".format(str(len(norm_next_points)), str(self.batch_size)))
            elif type(utility_type) == list:
                norm_next_points=[]
                if len(utility_type) == self.batch_size:
                    utility_type_dict = Counter(utility_type)
                    for utility_key, count in utility_type_dict.items():
                        if utility_key == 'ucb':
                            ucb_norm_next_points = self.suggest(self.ucb_utility,sampler=self.sampler,
                                        n_acqs=count) # return list
                            real_ucb_norm_next_points, ucb_trash_list = self._checkCandidateNumber(count, ucb_norm_next_points)
                            norm_next_points.extend(real_ucb_norm_next_points)
                        elif utility_key == 'ei':
                            ei_norm_next_points = self.suggest(self.ei_utility,sampler=self.sampler,
                                        n_acqs=count) # return list
                            real_ei_norm_next_points, ei_trash_list = self._checkCandidateNumber(count, ei_norm_next_points)
                            norm_next_points.extend(real_ei_norm_next_points)
                        elif utility_key == 'poi':
                            poi_norm_next_points = self.suggest(self.poi_utility,sampler=self.sampler,
                                        n_acqs=count) # return list
                            real_poi_norm_next_points, poi_trash_list = self._checkCandidateNumber(count, poi_norm_next_points)
                            norm_next_points.extend(real_poi_norm_next_points)
                        elif utility_key == 'es':
                            es_norm_next_points = self.suggest(self.es_utility,sampler=self.sampler,
                                        n_acqs=count) # return list
                            real_es_norm_next_points, es_trash_list = self._checkCandidateNumber(count, es_norm_next_points)
                            norm_next_points.extend(real_es_norm_next_points)
                else:
                    raise IndexError("Please fill utility list to match self.batch_size")
            else:
                raise TypeError("Please give string type or filled list")
               
        real_next_points=self._getRealCondition(norm_next_points)
        
        return real_next_points, norm_next_points

    def registerPoint(self, input_next_points, norm_input_next_points, property_list, input_result_list):
        """
        :param input_next_points (dict in list) : [{},{},{}] --> this list has sequence of condition which follow utility function
            ex) ['ucb', 'ucb', 'ucb', 'ucb', 'ei', 'ei', 'es', 'es']
        :param input_result_list (dict in list): [] --> include each result_dict, 
                                            this list has sequence of synthesis condition which called input_next_points
        :return : None
        """
        for process_idx, real_next_point in enumerate(input_next_points):
            optimal_value = input_result_list[process_idx]
            self._register(space=self._real_space, params=real_next_point, target=optimal_value)
        for process_idx, property_dict in enumerate(property_list):
            optimal_value = input_result_list[process_idx]
            self._property_space._keys=list(input_next_points[process_idx].keys())+list(property_dict.keys())
            self._register(space=self._property_space, params=list(input_next_points[process_idx].values())+list(property_dict.values()), target=optimal_value)
        for process_idx, norm_next_point in enumerate(norm_input_next_points):
            optimal_value = input_result_list[process_idx]
            self._register(space=self.space,params=norm_next_point, target=optimal_value)

    def output_space_realCondition(self, dirname, filename):
        """
        [Modified by HJ]

        Outputs complete space as csv file. --> convert normalize condition to real condition
        Simple function for testing
        
        Parameters
        ----------
        dirname (str) :"DB/2022XXXX
        filename : "{}_data" + .csv
        
        Returns
        -------
        None
        """
        total_path="{}/{}.csv".format(dirname, filename)
        if os.path.isdir(dirname) == False:
            os.makedirs(dirname)
        df = pd.DataFrame(data=self._real_space.params, columns=self._real_space.keys)
        df['Target'] = self._real_space.target
        df.to_csv(total_path, index=False)

    def output_space_property(self, dirname, filename):
        """
        [Modified by HJ]

        Outputs complete space as csv file. --> extract until all property based on real condition
        Simple function for testing
        
        Parameters
        ----------
        dirname (str) :"DB/2022XXXX
        filename : "{}_data" + .csv
        
        Returns
        -------
        None
        """
        total_path="{}/{}.csv".format(dirname, filename)
        if os.path.isdir(dirname) == False:
            os.makedirs(dirname)
        df = pd.DataFrame(data=self._property_space.params, columns=self._property_space.keys)
        df.to_csv(total_path, index=False)

    def closeModel(self,):
        """
        close model & print time table and maximum value
        
        Parameters
        ----------
        None
        
        Returns
        -------
        opt_msg, max_value_msg
        """
        if self.verbose:
            self.dispatch(Events.OPTMIZATION_END)
        
        self.process_end_time = time.time()
        
        opt_msg = "Time taken for {} optimizaiton: {:8.2f} seconds".format(self.sampler, self.process_end_time - self.process_start_time)
        max_value_msg = "Maximum value {:.3f} found in {} batches".format(self.max_val['val'], self.max_val['iter'])
        
        return opt_msg, max_value_msg

    def savedModel(self, directory_path, filename='bo_obj'):
        """
        save ML model to use already fitted model later.
        
        Arguments
        ---------
        directory_path (str)
        filename='bo_obj' (str) +.pickle
        
        Returns
        -------
        return None
        """
        fname = os.path.join(directory_path, filename+".pickle")
        if os.path.isdir(directory_path) == False:
            os.makedirs(directory_path)
        else:
            with open(fname, 'wb') as f:
                pickle.dump(self, f)
                time.sleep(3)
                
    def loadModel(self, directory_path, filename='optimizer'):
        """
        load ML model to use already fitted model later depending on filename.
        
        Arguments
        ---------
        directory_path (str)
        filename='optimizer' (str)
        
        Returns
        -------
        return loaded_model, model_obj
        """
        fname = os.path.join(directory_path, filename+".pickle")

        with open(fname, 'rb') as f:
            model_obj = pickle.load(f)

        return model_obj




