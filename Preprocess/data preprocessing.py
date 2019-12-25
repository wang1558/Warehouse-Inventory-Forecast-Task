# -*- coding: utf-8 -*-

import pandas as pd
import time
from time import strftime

##### Import dataset & filt

sales = pd.read_csv("""Path here""")
optimal_warehouse = pd.read_csv("""Path here""")

sales.rename(columns = {'ShipPostalCode':'destination_zip'}, inplace = True)
sales = sales[sales['Quantity']!=0]

na_records = sales[sales["destination_zip"].isna()]
clean_sales = sales[~sales["destination_zip"].isna()]
clean_sales = clean_sales[["destination_zip", "PurchaseDate", "ItemPrice", "ChannelProductId", "Quantity", "fulfillmentChannel", "master_sku"]]

##### Add Monthly and Weekly data column

clean_sales["PurchaseDate"] = pd.to_datetime(clean_sales["PurchaseDate"])
clean_sales["Year_Month"] = clean_sales["PurchaseDate"].dt.year.astype(str) + "-" + clean_sales["PurchaseDate"].dt.month.astype(str)

clean_sales["Weekly"] = clean_sales["PurchaseDate"].dt.year.astype(str) + "-" + clean_sales["PurchaseDate"].dt.month.astype(str) + '-' + clean_sales["PurchaseDate"].dt.day.astype(str)
d = clean_sales["Weekly"].apply(lambda x: time.strptime(x, "%Y-%m-%d"))
clean_sales["Week"] = d.apply(lambda x: int(strftime("%U", x)))
clean_sales["Week"][clean_sales["PurchaseDate"].dt.year.astype(str) == '2019'] = clean_sales['Week'][clean_sales["PurchaseDate"].dt.year.astype(str) == '2019'].apply(lambda x: x+53)

##### Revise Zipcode column for sales and warehouse dataset

clean_sales['destination_zip'] = clean_sales['destination_zip'].astype(int).astype(str)
clean_sales['destination_zip'] = clean_sales['destination_zip'].str[:5] # zipcode should only be 5 digits
clean_sales['destination_zip'] = clean_sales['destination_zip'].apply(lambda x : '0' * (5 - len(x)) + x)

optimal_warehouse['destination_zip'] = optimal_warehouse['destination_zip'].astype(str)
optimal_warehouse['warehouse_zip'] = optimal_warehouse['warehouse_zip'].astype(str)
optimal_warehouse['destination_zip'] = optimal_warehouse['destination_zip'].apply(lambda x : '0' * (5 - len(x)) + x)
optimal_warehouse['warehouse_zip'] = optimal_warehouse['warehouse_zip'].apply(lambda x : '0' * (5 - len(x)) + x)

##### Sum up sales

summarise_sales = clean_sales.groupby(['master_sku','destination_zip','Year_Month']).agg({
	'Quantity':{
	'units':'sum'
}
}).reset_index()

summarise_sales_weekly = clean_sales.groupby(['master_sku','destination_zip','Week']).agg({
	'Quantity':{
	'units':'sum'
}
}).reset_index()

# get the warehouse assigment per customer zip code and product
warehouse_demand = pd.merge(summarise_sales,optimal_warehouse,how="left",on=["master_sku","destination_zip"])
warehouse_demand_weekly = pd.merge(summarise_sales_weekly,optimal_warehouse,how="left",on=["master_sku","destination_zip"])

warehouse_demand = warehouse_demand[~warehouse_demand['warehouse_zip'].isna()]
warehouse_demand_weekly = warehouse_demand_weekly[~warehouse_demand_weekly['warehouse_zip'].isna()]

warehouse_demand = warehouse_demand[['master_sku', 'destination_zip', ('Year_Month', ''), ('Quantity', 'units'), 'warehouse_zip']]
warehouse_demand_weekly = warehouse_demand_weekly[['master_sku', 'destination_zip', ('Week', ''), ('Quantity', 'units'), 'warehouse_zip']]

warehouse_demand.rename(columns = {('Year_Month', ''):'Year_Month', ('Quantity', 'units'):'Quantity'}, inplace = True)
warehouse_demand_weekly.rename(columns = {('Week', ''):'Week', ('Quantity', 'units'):'Quantity'}, inplace = True)

summarise_demand = warehouse_demand.groupby(['master_sku','warehouse_zip','Year_Month']).agg({
	'Quantity':{
	'units':'sum'
}
}).reset_index()

summarise_demand_weekly = warehouse_demand_weekly.groupby(['master_sku','warehouse_zip','Week']).agg({
	'Quantity':{
	'units':'sum'
}
}).reset_index()


#summarise_demand.rename(columns = {('master_sku', ''):'master_sku', ('warehouse_zip', ''):'warehouse_zip',
#                                   ('Year_Month', ''):'Year_Month', ('Quantity', 'units'):'Quantity'}, inplace = True)
summarise_demand.columns = ['master_sku','warehouse_zip','Year_Month','Quantity']
summarise_demand_weekly.columns = ['master_sku','warehouse_zip','Week','Quantity']

summarise_total = warehouse_demand.groupby(['master_sku','Year_Month']).agg({
	'Quantity':{
	'units':'sum'
}
}).reset_index()

summarise_total_weekly = warehouse_demand_weekly.groupby(['master_sku','Week']).agg({
	'Quantity':{
	'units':'sum'
}
}).reset_index()

#summarise_total.rename(columns = {('master_sku', ''):'master_sku', ('Year_Month', ''):'Year_Month', ('Quantity', 'units'):'Quantity'}, inplace = True)
summarise_total.columns = ['master_sku','Year_Month','Quantity']
summarise_total_weekly.columns = ['master_sku','Week','Quantity']


warehouse_demand_percentage = pd.merge(summarise_demand,summarise_total,how="left",on=["master_sku","Year_Month"])
warehouse_demand_weekly_percentage = pd.merge(summarise_demand_weekly,summarise_total_weekly,how="left",on=["master_sku","Week"])

warehouse_demand_percentage['Percentage'] = warehouse_demand_percentage['Quantity_x']/warehouse_demand_percentage['Quantity_y']
warehouse_demand_weekly_percentage['Percentage'] = warehouse_demand_weekly_percentage['Quantity_x']/warehouse_demand_weekly_percentage['Quantity_y']
