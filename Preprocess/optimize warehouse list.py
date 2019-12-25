# -*- coding: utf-8 -*-

from csv import DictReader
import os
import pandas as pd
import numpy as np

def load_csv(fpath):
    f = DictReader(open(fpath,"r"))
    return list(f)


def clean_zip(dirty_zip):
    if "-" in dirty_zip:
        dirty_zip = dirty_zip.split("-")[0]
    return '0' * (5 - len(dirty_zip)) + dirty_zip
#INPUT:  {"ups":["input/ups/02136.csv"],"fedex":[]}
#OUT:  {"10512":{"ups":[{"warehouse":"19211","vendor":"ups","zone":2,"ground_days":1}],"fedex":[{"warehouse":"19211","vendor":"fedex","zone":2,"ground_days":1}]}}
def build_zip_to_warehouse_map(vendor_paths):
    zip_to_warehouse_map = {}
    for vendor in vendor_paths:
        for warehouse_file_path in vendor_paths[vendor]:
            with open(warehouse_file_path,"r",encoding='utf-8-sig') as csvfile:
                for row in DictReader(csvfile):
                    row["destination_zip"] = clean_zip(row["destination_zip"])
                    row["warehouse_zip"] = clean_zip(row["warehouse_zip"])
                    if row["destination_zip"] not in  zip_to_warehouse_map:
                        zip_to_warehouse_map[row["destination_zip"]] = {}
                    if vendor not in  zip_to_warehouse_map[row["destination_zip"]]:
                        zip_to_warehouse_map[row["destination_zip"]][vendor] = []

                    zip_to_warehouse_map[row["destination_zip"]][vendor].append(row)

    return zip_to_warehouse_map

def prime_fedex(zipdata):
    one_day = pd.DataFrame()
    for zipcode in zipdata.keys():
        optimize_fedex = {}
        warehouse_list = []
        zone_list = []
        if len(zipdata[zipcode]) < 2:
            pass
        else:
            for i in range(len(zipdata[zipcode]['fedex'])):
                if list(zipdata[zipcode]['fedex'][i].values())[3] == '1':
                    warehouse_list.append(list(zipdata[zipcode]['fedex'][i].values())[0])
                    zone_list.append(list(zipdata[zipcode]['fedex'][i].values())[2])
                    optimize_fedex.update({'destination': zipcode, 'warehouse': warehouse_list, 'zone': zone_list})
                else:
                    pass
        if len(optimize_fedex) > 0:
            df = pd.DataFrame([optimize_fedex], columns=optimize_fedex.keys())
            one_day = pd.concat([one_day, df])
    
    return one_day

def prime_ups(zipdata):
    one_day = pd.DataFrame()
    for zipcode in zipdata.keys():
        optimize_ups = {}
        warehouse_list = []
        zone_list = []
        for i in range(len(zipdata[zipcode]['ups'])):
            if list(zipdata[zipcode]['ups'][i].values())[3] == '1':
                warehouse_list.append(list(zipdata[zipcode]['ups'][i].values())[0])
                zone_list.append(list(zipdata[zipcode]['ups'][i].values())[2])
                optimize_ups.update({'destination': zipcode, 'warehouse': warehouse_list, 'zone': zone_list})
            else:
                pass
        if len(optimize_ups) > 0:
            df = pd.DataFrame([optimize_ups], columns=optimize_ups.keys())
            one_day = pd.concat([one_day, df])
    
    return one_day

def twodays_fedex(zipdata):
    two_days = pd.DataFrame()
    for zipcode in zipdata.keys():
        optimize_fedex = {}
        warehouse_list = []
        zone_list = []
        if len(zipdata[zipcode]) < 2:
            pass
        else:
            for i in range(len(zipdata[zipcode]['fedex'])):
                if list(zipdata[zipcode]['fedex'][i].values())[3] == '1':
                    break
                elif list(zipdata[zipcode]['fedex'][i].values())[3] == '2':
                    warehouse_list.append(list(zipdata[zipcode]['fedex'][i].values())[0])
                    zone_list.append(list(zipdata[zipcode]['fedex'][i].values())[2])
                    optimize_fedex.update({'destination': zipcode, 'warehouse': warehouse_list, 'zone': zone_list})
                else:
                    pass
        if len(optimize_fedex) > 0:
            df = pd.DataFrame([optimize_fedex], columns=optimize_fedex.keys())
            two_days = pd.concat([two_days, df])
    
    one_day = prime_fedex(zipdata)
    for j in two_days['destination']:
        if j in list(one_day['destination']):
            two_days = two_days[two_days['destination'] != j]
    
    return two_days

def twodays_ups(zipdata):
    two_days = pd.DataFrame()
    for zipcode in zipdata.keys():
        optimize_ups = {}
        warehouse_list = []
        zone_list = []
        for i in range(len(zipdata[zipcode]['ups'])):
            if list(zipdata[zipcode]['ups'][i].values())[3] == '1':
                break
            elif list(zipdata[zipcode]['ups'][i].values())[3] == '2':
                warehouse_list.append(list(zipdata[zipcode]['ups'][i].values())[0])
                zone_list.append(list(zipdata[zipcode]['ups'][i].values())[2])
                optimize_ups.update({'destination': zipcode, 'warehouse': warehouse_list, 'zone': zone_list})
            else:
                pass
        if len(optimize_ups) > 0:
            df = pd.DataFrame([optimize_ups], columns=optimize_ups.keys())
            two_days = pd.concat([two_days, df])
    
    one_day = prime_ups(zipdata)
    for j in two_days['destination']:
        if j in list(one_day['destination']):
            two_days = two_days[two_days['destination'] != j]
    
    return two_days

def moredays_fedex(zipdata):
    more_days = pd.DataFrame()
    for zipcode in zipdata.keys():
        optimize_fedex = {}
        warehouse_list = []
        zone_list = []
        if len(zipdata[zipcode]) < 2:
            pass
        else:
            for i in range(len(zipdata[zipcode]['fedex'])):
                if list(zipdata[zipcode]['fedex'][i].values())[3] == '1':
                    break
                elif list(zipdata[zipcode]['fedex'][i].values())[3] == '2':
                    break
                else:
                    warehouse_list.append(list(zipdata[zipcode]['fedex'][i].values())[0])
                    zone_list.append(list(zipdata[zipcode]['fedex'][i].values())[2])
                    optimize_fedex.update({'destination': zipcode, 'warehouse': warehouse_list, 'zone': zone_list})
        if len(optimize_fedex) > 0:
            df = pd.DataFrame([optimize_fedex], columns=optimize_fedex.keys())
            more_days = pd.concat([more_days, df])
    
    one_day = prime_fedex(zipdata)
    two_days = twodays_fedex(zipdata)
    for j in more_days['destination']:
        if j in list(one_day['destination']):
            more_days = more_days[more_days['destination'] != j]
        elif j in list(two_days['destination']):
            more_days = more_days[more_days['destination'] != j]
    
    return more_days

def moredays_ups(zipdata):
    more_days = pd.DataFrame()
    for zipcode in zipdata.keys():
        optimize_ups = {}
        warehouse_list = []
        zone_list = []
        for i in range(len(zipdata[zipcode]['ups'])):
            if list(zipdata[zipcode]['ups'][i].values())[3] == '1':
                break
            elif list(zipdata[zipcode]['ups'][i].values())[3] == '2':
                break
            else:
                warehouse_list.append(list(zipdata[zipcode]['ups'][i].values())[0])
                zone_list.append(list(zipdata[zipcode]['ups'][i].values())[2])
                optimize_ups.update({'destination': zipcode, 'warehouse': warehouse_list, 'zone': zone_list})
        if len(optimize_ups) > 0:
            df = pd.DataFrame([optimize_ups], columns=optimize_ups.keys())
            more_days = pd.concat([more_days, df])
    
    one_day = prime_ups(zipdata)
    two_days = twodays_ups(zipdata)
    for j in more_days['destination']:
        if j in list(one_day['destination']):
            more_days = more_days[more_days['destination'] != j]
        elif j in list(two_days['destination']):
            more_days = more_days[more_days['destination'] != j]
    
    return more_days

def find_the_optimization(dataframe):
    optimize = pd.DataFrame()
    warehouse = []
    zone = []
    for i in range(len(dataframe.zone)):
        index = np.argmin(list(dataframe.zone)[i])
        warehouse.append(list(dataframe.warehouse)[i][index])
        zone.append(list(dataframe.zone)[i][index])
    
    optimize['destination'] = list(dataframe.destination)    
    optimize['warehouse'] = warehouse
    optimize['zone'] = zone
    
    return optimize


if __name__ == '__main__':
    
    directory_fedex = """Path here"""
    filelist_fedex = []
    for filename in os.listdir(directory_fedex):
        if filename.endswith(".csv"): 
            file = directory_fedex + '/' + filename
            filelist_fedex = filelist_fedex + [file]

    directory_ups = """Path here"""
    filelist_ups = []
    for filename in os.listdir(directory_ups):
        if filename.endswith(".csv"): 
            file = directory_ups + '/' + filename
            filelist_ups = filelist_ups + [file]            
            
    indexer = build_zip_to_warehouse_map({"fedex":filelist_fedex[:-1], "ups":filelist_ups[:-1]})
    a = 1
         
    ##### Select all warehouses which are able to ship within one day and only two days (exclusive)
    
    ups_oneday = prime_ups(indexer)
    fedex_oneday = prime_fedex(indexer)
    
    ups_twodays = twodays_ups(indexer)
    fedex_twodays = twodays_fedex(indexer)
    
    rate_ups = pd.read_csv(filelist_ups[-1])
    rate_fedex = pd.read_csv(filelist_fedex[-1])
    
    ###### Select one of the warehouse with the smallest cost (only depending on zone if weights are the same)
    
    optimize_oneday_ups = find_the_optimization(ups_oneday)
    optimize_oneday_fedex = find_the_optimization(fedex_oneday)
    optimize_twodays_ups = find_the_optimization(ups_twodays)
    optimize_twodays_fedex = find_the_optimization(fedex_twodays)
    
    ##### Comparison bewteen ups and fedex
    
    optimization_oneday = pd.merge(optimize_oneday_fedex,optimize_oneday_ups,how="outer",on=['destination'])
    optimization_oneday.rename(columns = {'warehouse_x':'warehouse_fedex', 'zone_x':'zone_fedex',
                                          'warehouse_y':'warehouse_ups', 'zone_y':'zone_ups'}, inplace = True)
    
    optimization_twodays = pd.merge(optimize_twodays_fedex,optimize_twodays_ups,how="outer",on=['destination'])
    optimization_twodays.rename(columns = {'warehouse_x':'warehouse_fedex', 'zone_x':'zone_fedex',
                                           'warehouse_y':'warehouse_ups', 'zone_y':'zone_ups'}, inplace = True)

    rate_comparison = pd.merge(rate_fedex,rate_ups,how="outer",on=['lbs'])
    
    difference = rate_fedex.set_index('lbs').subtract(rate_ups.set_index('lbs'))
    # Difference shows that when both fedex and ups have the same zone warehouse, always choose ups
    
    ##### Based on above data, the optimization should focus on optimal warehouses with different zones
    
    oneday_demand = optimization_oneday[optimization_oneday['zone_fedex'] != optimization_oneday['zone_ups']]
    #twodays_demand = optimization_twodays[optimization_twodays['zone_fedex'] != optimization_twodays['zone_ups']]
    
    oneday_easychoice = optimization_oneday[optimization_oneday['zone_fedex'] == optimization_oneday['zone_ups']]
    oneday_ups_choice = oneday_demand[oneday_demand['zone_fedex'].isna()]
    oneday_fedex_choice = oneday_demand[oneday_demand['zone_ups'].isna()]
    
    oneday_ups_choice = pd.concat([oneday_ups_choice, oneday_easychoice])
    
    ##### Hard choice requires weights comparison
    
    oneday_hardchoice = oneday_demand[~oneday_demand['zone_fedex'].isna()]
    oneday_hardchoice = oneday_hardchoice[~oneday_hardchoice['zone_ups'].isna()]
    
    difference_2to3 = rate_fedex.set_index('lbs')['2'].subtract(rate_ups.set_index('lbs')['3'])
    difference_2to4 = rate_fedex.set_index('lbs')['2'].subtract(rate_ups.set_index('lbs')['4'])
    difference_3to2 = rate_fedex.set_index('lbs')['3'].subtract(rate_ups.set_index('lbs')['2'])
    difference_3to4 = rate_fedex.set_index('lbs')['3'].subtract(rate_ups.set_index('lbs')['4'])
    difference_4to2 = rate_fedex.set_index('lbs')['4'].subtract(rate_ups.set_index('lbs')['2'])
    difference_4to3 = rate_fedex.set_index('lbs')['4'].subtract(rate_ups.set_index('lbs')['3']) 
    difference_4to5 = rate_fedex.set_index('lbs')['4'].subtract(rate_ups.set_index('lbs')['5']) 
    
    oneday_ups_choice = pd.concat([oneday_ups_choice, oneday_hardchoice[oneday_hardchoice['zone_ups'].astype(int)<oneday_hardchoice['zone_fedex'].astype(int)]])
    oneday_hardchoice = oneday_hardchoice[oneday_hardchoice['zone_ups'].astype(int)>oneday_hardchoice['zone_fedex'].astype(int)]
    
    oneday_ups_choice = oneday_ups_choice[['destination','warehouse_ups','zone_ups']]
    oneday_fedex_choice = oneday_fedex_choice[['destination','warehouse_fedex','zone_fedex']]
    
    ##### Dealing with two days shipping
    
    index = []
    for i in optimization_twodays['destination']:
        if i not in list(optimization_oneday['destination']):
            index.append(i)
    
    # drop rows shown in oneday file
    optimization_twodays = optimization_twodays[optimization_twodays['destination'].isin(index)]
    twodays_demand = optimization_twodays[optimization_twodays['zone_fedex'] != optimization_twodays['zone_ups']]
    
    twodays_easychoice = optimization_twodays[optimization_twodays['zone_fedex'] == optimization_twodays['zone_ups']]
    twodays_ups_choice = twodays_demand[twodays_demand['zone_fedex'].isna()]
    twodays_fedex_choice = twodays_demand[twodays_demand['zone_ups'].isna()]
    
    twodays_ups_choice = pd.concat([twodays_ups_choice, twodays_easychoice])

    twodays_hardchoice = twodays_demand[~twodays_demand['zone_fedex'].isna()]
    twodays_hardchoice = twodays_hardchoice[~twodays_hardchoice['zone_ups'].isna()]
    
    twodays_ups_choice = pd.concat([twodays_ups_choice, twodays_hardchoice[twodays_hardchoice['zone_ups'].astype(int)<twodays_hardchoice['zone_fedex'].astype(int)]])
    twodays_hardchoice = twodays_hardchoice[twodays_hardchoice['zone_ups'].astype(int)>twodays_hardchoice['zone_fedex'].astype(int)]
    
    twodays_ups_choice = twodays_ups_choice[['destination','warehouse_ups','zone_ups']]
       
    ##### Case for more than two days
    
    ups_moredays = moredays_ups(indexer)
    fedex_moredays = moredays_fedex(indexer)
    
    optimize_moredays_ups = find_the_optimization(ups_moredays)
    optimize_moredays_fedex = find_the_optimization(fedex_moredays)
    
    optimization_moredays = pd.merge(optimize_moredays_fedex,optimize_moredays_ups,how="outer",on=['destination'])
    optimization_moredays.rename(columns = {'warehouse_x':'warehouse_fedex', 'zone_x':'zone_fedex',
                                            'warehouse_y':'warehouse_ups', 'zone_y':'zone_ups'}, inplace = True)
    
    
    