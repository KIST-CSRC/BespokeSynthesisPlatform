#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ##
# @brief    [Fitness] UV target (lambdamax, FWHM, intensity) file
# @author   Hyuk Jun Yoo (yoohj9475@kist.re.kr)   
# @version  1_2   
# TEST 2021-11-01
# TEST 2022-04-11

def lambdamaxFitness(result_dict, target_condition_dict):
    """
    calculate Fitness value for targeting lambdamax (this function has 550nm lambdamax target)

    :param result_dict (dict): {"GetUVdata":{"Wavelength":[...],"RawSpectrum":[...],"Property":{'lambdamax': [300.214759], 'FWHM': [549.221933]}}}
    :param target_condition_dict (dict): {"GetUVdata":{"Property":{"lambdamax":500,"FWHM":100},"Ratio":{"lambdamax":0.8,"FWHM":0.2}}}

    :return optimal_value (float): optimal_value
    """

    for key, each_action_target_condition_dict in target_condition_dict.items():
        optimal_value=0
        # key --> "GetUVdata"
        # each_action_target_condition_dict --> {"Property":{"lambdamax":500,"FWHM":100},"Ratio":{"lambdamax":0.8,"FWHM":0.2}}
        target_type_list = list(each_action_target_condition_dict["Property"].keys()) # target_type_list --> ["lambdamax", "FWHM"]
        target_value_list = list(each_action_target_condition_dict["Property"].values()) # target_value_list --> [500, 100]
        target_optimal_ratio_list = list(each_action_target_condition_dict["Ratio"].values()) # [0.8, 0.2]
        
        for idx, each_target_type in enumerate(target_type_list): 
            # idx --> 0, 1 (lambdamax or FWHM)
            # each_target_type --> "lambdamax", "FWHM"
            if len(result_dict[key]["Property"][each_target_type])==0:
                optimal_value += -1*target_optimal_ratio_list[idx]
            else:
                scaling_factor_left = result_dict[key]["Property"][each_target_type][0]-300
                scaling_factor_right = 800-result_dict[key]["Property"][each_target_type][0]
                if scaling_factor_left>scaling_factor_right:
                    optimal_value += -abs(target_value_list[idx]-result_dict[key]["Property"][each_target_type][0])/(scaling_factor_left)*target_optimal_ratio_list[idx]
                else:
                    optimal_value += -abs(target_value_list[idx]-result_dict[key]["Property"][each_target_type][0])/(scaling_factor_right)*target_optimal_ratio_list[idx]

    return optimal_value


def lambdamaxFWHMFitness(result_dict, target_condition_dict):
    """
    calculate Fitness value

    :param result_dict (dict): {"GetUVdata":{"Wavelength":[...],"RawSpectrum":[...],"Property":{'lambdamax': [300.214759], 'FWHM': [549.221933]}}}
    :param target_condition_dict (dict): {"GetUVdata":{"Property":{"lambdamax":500,"FWHM":100},"Ratio":{"lambdamax":0.8,"FWHM":0.2}}}

    :return optimal_value (float) and property_tuple (tuple): optimal_value
    """
    maxIdx=0
    for key, each_action_target_condition_dict in target_condition_dict.items():
        optimal_value=0
        # key --> "GetUVdata"
        # each_action_target_condition_dict --> {"Property":{"lambdamax":500,"FWHM":100},"Ratio":{"lambdamax":0.8,"FWHM":0.2}}
        target_type_list = list(each_action_target_condition_dict["Property"].keys()) # target_type_list --> ["lambdamax", "FWHM"]
        target_value_list = list(each_action_target_condition_dict["Property"].values()) # target_value_list --> [500, 100]
        target_optimal_ratio_list = list(each_action_target_condition_dict["Ratio"].values()) # [0.8, 0.2]
        lambdamax_single=None
        intensity_single=None
        FWHM_single=None
        for target_idx, each_target_type in enumerate(target_type_list): # property 별로 Fitness 계산하고, optimal_value에 통합
            # target_idx --> 0, 1 (lambdamax or FWHM)
            # each_target_type --> "lambdamax", "FWHM"
            if len(result_dict[key]["Property"][each_target_type])==0:
                optimal_value += -1*target_optimal_ratio_list[target_idx]
                lambdamax_single=0
                intensity_single=0
                FWHM_single=0
            else:
                Intensity_peaks_list=result_dict[key]["Property"]["intensity"]
                maxIdx = Intensity_peaks_list.index(max(Intensity_peaks_list))
                if target_idx == 0:
                    lambdamax_scaling_factor_left = result_dict[key]["Property"][each_target_type][maxIdx]-300
                    lambdamax_scaling_factor_right = 800-result_dict[key]["Property"][each_target_type][maxIdx]

                    if lambdamax_scaling_factor_left>lambdamax_scaling_factor_right:
                        # lambdamax_Fitness = -abs(target_value_list[maxIdx]-result_dict[key]["Property"][each_target_type][maxIdx])/(lambdamax_scaling_factor_left)*target_optimal_ratio_list[target_idx]
                        # FWHM_Fitness = -abs(target_value_list[target_idx]-result_dict[key]["Property"][each_target_type][maxIdx])
                        optimal_value += -abs(target_value_list[target_idx]-result_dict[key]["Property"][each_target_type][maxIdx])/(lambdamax_scaling_factor_left)*target_optimal_ratio_list[target_idx]
                    else:
                        optimal_value += -abs(target_value_list[target_idx]-result_dict[key]["Property"][each_target_type][maxIdx])/(lambdamax_scaling_factor_right)*target_optimal_ratio_list[target_idx]
                else:
                    FWHM_scaling_factor = 500
                    optimal_value += -abs(result_dict[key]["Property"][each_target_type][maxIdx])/(FWHM_scaling_factor)*target_optimal_ratio_list[target_idx]
                lambdamax_single=result_dict["GetUVdata"]["Property"]["lambdamax"][maxIdx]
                intensity_single=result_dict["GetUVdata"]["Property"]["intensity"][maxIdx]
                FWHM_single=result_dict["GetUVdata"]["Property"]["FWHM"][maxIdx]
    property_tuple=(lambdamax_single, intensity_single, FWHM_single)
    
    return optimal_value, property_tuple

def lambdamaxFWHMintensityFitness(result_dict, target_condition_dict):
    """
    calculate Fitness value

    :param result_dict (dict): {"GetUVdata":{"Wavelength":[...],"RawSpectrum":[...],"Property":{'lambdamax': 667.901297, 'intensity': 0.754869663, 'FWHM': 252.874914}}}
    :param target_condition_dict (dict): {"GetUVdata":{"Property":{"lambdamax":500},"Ratio":{"lambdamax":0.9,"FWHM":0.03, "intenisty":0.07}}}

    :return optimal_value (float) and property_tuple (tuple): optimal_value
    :return total_property_dict (dict)
    """
    total_property_dict={}
    for key, each_action_target_condition_dict in target_condition_dict.items():
        optimal_value=0
        # key --> "GetUVdata"
        # each_action_target_condition_dict --> {"Property":{"lambdamax":500,"FWHM":100},"Ratio":{"lambdamax":0.8,"FWHM":0.2}}
        target_type_list = list(each_action_target_condition_dict["Ratio"].keys()) # target_type_list --> ["lambdamax"]
        target_value_list = list(each_action_target_condition_dict["Property"].values()) # target_value_list --> [500]
        target_optimal_ratio_list = list(each_action_target_condition_dict["Ratio"].values()) # [0.9, 0.03, 0.07]
        for target_idx, each_target_type in enumerate(target_type_list):
            if result_dict[key]["Property"][each_target_type]==0:
                optimal_value += -1*target_optimal_ratio_list[target_idx]
                optimal_value=float(optimal_value)
            else:
                if each_target_type == "lambdamax":
                    lambdamax_scaling_factor_left = target_value_list[target_idx]-300
                    lambdamax_scaling_factor_right = 850-target_value_list[target_idx]

                    if lambdamax_scaling_factor_left>lambdamax_scaling_factor_right:
                        optimal_value -= abs(target_value_list[target_idx]-result_dict[key]["Property"][each_target_type])/(lambdamax_scaling_factor_left)*target_optimal_ratio_list[target_idx]
                    else:
                        optimal_value -= abs(target_value_list[target_idx]-result_dict[key]["Property"][each_target_type])/(lambdamax_scaling_factor_right)*target_optimal_ratio_list[target_idx]
                elif each_target_type == "FWHM":
                    FWHM_scaling_factor = 550
                    optimal_value -= abs(result_dict[key]["Property"][each_target_type])/(FWHM_scaling_factor)*target_optimal_ratio_list[target_idx]
                elif each_target_type == "intensity":
                    optimal_value -= abs(1-result_dict[key]["Property"][each_target_type])*target_optimal_ratio_list[target_idx]
            total_property_dict[each_target_type]=result_dict[key]["Property"][each_target_type]

    return optimal_value, total_property_dict

