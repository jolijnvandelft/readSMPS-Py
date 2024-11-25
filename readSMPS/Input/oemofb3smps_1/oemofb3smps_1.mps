NAME oemofb3smps_1
ROWS
 N  OBJ
 L  c_u_GenericInvestmentStorageBlock_init_content_limit(BB_heat_decentral_storage)_
 L  c_u_GenericInvestmentStorageBlock_init_content_limit(B_h2_cavern)_
 L  c_u_GenericInvestmentStorageBlock_init_content_limit(B_heat_decentral_storage)_
 L  c_u_GenericInvestmentStorageBlock_init_content_limit(BB_h2_cavern)_
 L  c_u_GenericInvestmentStorageBlock_init_content_limit(B_electricity_liion_battery)_
 L  c_u_GenericInvestmentStorageBlock_init_content_limit(BB_heat_central_storage)_
 L  c_u_GenericInvestmentStorageBlock_init_content_limit(BB_electricity_liion_battery)_
 L  c_u_GenericInvestmentStorageBlock_init_content_limit(B_heat_central_storage)_
 E  c_e_GenericInvestmentStorageBlock_power_coupled(BB_heat_decentral_storage_0)_
 E  c_e_GenericInvestmentStorageBlock_power_coupled(B_h2_cavern_0)_
 E  c_e_GenericInvestmentStorageBlock_power_coupled(B_heat_decentral_storage_0)_
 E  c_e_GenericInvestmentStorageBlock_power_coupled(BB_h2_cavern_0)_
 E  c_e_GenericInvestmentStorageBlock_power_coupled(B_electricity_liion_battery_0)_
 E  c_e_GenericInvestmentStorageBlock_power_coupled(BB_heat_central_storage_0)_
 E  c_e_GenericInvestmentStorageBlock_power_coupled(BB_electricity_liion_battery_0)_
 E  c_e_GenericInvestmentStorageBlock_power_coupled(B_heat_central_storage_0)_
 L  c_u_integral_limit_emission_factor_constraint_
 E  c_e_equate_flows_heat_central_B(0_0)_
 E  c_e_equate_flows_heat_central_B(0_1)_
 E  c_e_equate_flows_heat_central_B(0_2)_
 E  c_e_equate_flows_heat_central_BB(0_0)_
 E  c_e_equate_flows_heat_central_BB(0_1)_
 E  c_e_equate_flows_heat_central_BB(0_2)_
 E  c_e_equate_flows_heat_decentral_B(0_0)_
 E  c_e_equate_flows_heat_decentral_B(0_1)_
 E  c_e_equate_flows_heat_decentral_B(0_2)_
 E  c_e_equate_flows_heat_decentral_BB(0_0)_
 E  c_e_equate_flows_heat_decentral_BB(0_1)_
 E  c_e_equate_flows_heat_decentral_BB(0_2)_
 E  c_e_BusBlock_balance(B_ch4_0_0)_
 E  c_e_BusBlock_balance(B_ch4_0_1)_
 E  c_e_BusBlock_balance(B_ch4_0_2)_
 E  c_e_BusBlock_balance(BB_heat_decentral_0_0)_
 E  c_e_BusBlock_balance(BB_heat_decentral_0_1)_
 E  c_e_BusBlock_balance(BB_heat_decentral_0_2)_
 E  c_e_BusBlock_balance(B_heat_central_0_0)_
 E  c_e_BusBlock_balance(B_heat_central_0_1)_
 E  c_e_BusBlock_balance(B_heat_central_0_2)_
 E  c_e_BusBlock_balance(BB_ch4_0_0)_
 E  c_e_BusBlock_balance(BB_ch4_0_1)_
 E  c_e_BusBlock_balance(BB_ch4_0_2)_
 E  c_e_BusBlock_balance(B_h2_0_0)_
 E  c_e_BusBlock_balance(B_h2_0_1)_
 E  c_e_BusBlock_balance(B_h2_0_2)_
 E  c_e_BusBlock_balance(B_heat_decentral_0_0)_
 E  c_e_BusBlock_balance(B_heat_decentral_0_1)_
 E  c_e_BusBlock_balance(B_heat_decentral_0_2)_
 E  c_e_BusBlock_balance(BB_heat_central_0_0)_
 E  c_e_BusBlock_balance(BB_heat_central_0_1)_
 E  c_e_BusBlock_balance(BB_heat_central_0_2)_
 E  c_e_BusBlock_balance(BB_h2_0_0)_
 E  c_e_BusBlock_balance(BB_h2_0_1)_
 E  c_e_BusBlock_balance(BB_h2_0_2)_
 E  c_e_BusBlock_balance(BB_electricity_0_0)_
 E  c_e_BusBlock_balance(BB_electricity_0_1)_
 E  c_e_BusBlock_balance(BB_electricity_0_2)_
 E  c_e_BusBlock_balance(B_electricity_0_0)_
 E  c_e_BusBlock_balance(B_electricity_0_1)_
 E  c_e_BusBlock_balance(B_electricity_0_2)_
COLUMNS
    InvestmentFlowBlock_invest(B_ch4_bpchp_B_electricity_0)  OBJ       9.6413159065100044e+04
    InvestmentFlowBlock_invest(B_ch4_gt_B_electricity_0)  OBJ       5.1286220648956383e+04
    InvestmentFlowBlock_invest(B_solar_pv_B_electricity_0)  OBJ       4.1235033167630216e+04
    InvestmentFlowBlock_invest(BB_h2_BB_h2_cavern_0)  OBJ       6.6517040301476463e+03
    InvestmentFlowBlock_invest(BB_h2_BB_h2_cavern_0)  c_e_GenericInvestmentStorageBlock_power_coupled(BB_h2_cavern_0)_  -1
    InvestmentFlowBlock_invest(BB_h2_bpchp_BB_electricity_0)  OBJ       9.6413159065100044e+04
    InvestmentFlowBlock_invest(B_electricity_heatpump_large_B_heat_central_0)  OBJ       3.8486818788279110e+04
    InvestmentFlowBlock_invest(BB_ch4_boiler_large_BB_heat_central_0)  OBJ       4.9005981393227294e+03
    InvestmentFlowBlock_invest(BB_electricity_BB_electricity_liion_battery_0)  OBJ       4.0098059480196794e+03
    InvestmentFlowBlock_invest(BB_electricity_BB_electricity_liion_battery_0)  c_e_GenericInvestmentStorageBlock_power_coupled(BB_electricity_liion_battery_0)_  -1
    InvestmentFlowBlock_invest(B_wind_onshore_B_electricity_0)  OBJ       6.6856895168314863e+04
    InvestmentFlowBlock_invest(B_electricity_pth_B_heat_central_0)  OBJ       5.3349050197177312e+03
    InvestmentFlowBlock_invest(B_ch4_boiler_large_B_heat_central_0)  OBJ       4.9005981393227294e+03
    InvestmentFlowBlock_invest(B_electricity_heatpump_small_B_heat_decentral_0)  OBJ       8.5215551622390965e+04
    InvestmentFlowBlock_invest(B_electricity_electrolyzer_B_h2_0)  OBJ       4.8336118297629037e+04
    InvestmentFlowBlock_invest(BB_electricity_heatpump_small_BB_heat_decentral_0)  OBJ       8.5215551622390965e+04
    InvestmentFlowBlock_invest(B_h2_B_h2_cavern_0)  OBJ       6.6517040301476463e+03
    InvestmentFlowBlock_invest(B_h2_B_h2_cavern_0)  c_e_GenericInvestmentStorageBlock_power_coupled(B_h2_cavern_0)_  -1
    InvestmentFlowBlock_invest(B_ch4_boiler_small_B_heat_decentral_0)  OBJ       2.9802620078870925e+04
    InvestmentFlowBlock_invest(BB_electricity_electrolyzer_BB_h2_0)  OBJ       4.8336118297629037e+04
    InvestmentFlowBlock_invest(BB_solar_pv_BB_electricity_0)  OBJ       2.4520368923681439e+04
    InvestmentFlowBlock_invest(BB_ch4_bpchp_BB_electricity_0)  OBJ       9.6413159065100044e+04
    InvestmentFlowBlock_invest(BB_hydro_ror_BB_electricity_0)  OBJ       1.9260553536984959e+05
    InvestmentFlowBlock_invest(BB_electricity_pth_BB_heat_central_0)  OBJ       5.3349050197177312e+03
    InvestmentFlowBlock_invest(BB_electricity_heatpump_large_BB_heat_central_0)  OBJ       3.8486818788279110e+04
    InvestmentFlowBlock_invest(B_electricity_B_electricity_liion_battery_0)  OBJ       4.0098059480196794e+03
    InvestmentFlowBlock_invest(B_electricity_B_electricity_liion_battery_0)  c_e_GenericInvestmentStorageBlock_power_coupled(B_electricity_liion_battery_0)_  -1
    InvestmentFlowBlock_invest(B_h2_gt_B_electricity_0)  OBJ       5.1286220648956383e+04
    InvestmentFlowBlock_invest(B_h2_bpchp_B_electricity_0)  OBJ       9.6413159065100044e+04
    InvestmentFlowBlock_invest(BB_ch4_gt_BB_electricity_0)  OBJ       5.1286220648956383e+04
    InvestmentFlowBlock_invest(BB_ch4_boiler_small_BB_heat_decentral_0)  OBJ       2.9802620078870925e+04
    InvestmentFlowBlock_invest(BB_wind_onshore_BB_electricity_0)  OBJ       6.6856895168314863e+04
    InvestmentFlowBlock_invest(BB_h2_gt_BB_electricity_0)  OBJ       5.1286220648956383e+04
    InvestmentFlowBlock_invest(B_hydro_ror_B_electricity_0)  OBJ       1.9260553536984959e+05
    flow(B_ch4_boiler_large_B_heat_central_0_0)  OBJ       1
    flow(B_ch4_boiler_large_B_heat_central_0_0)  c_e_equate_flows_heat_central_B(0_0)_  0.69
    flow(B_ch4_boiler_large_B_heat_central_0_0)  c_e_BusBlock_balance(B_heat_central_0_0)_  1
    flow(B_ch4_boiler_large_B_heat_central_0_1)  OBJ       1
    flow(B_ch4_boiler_large_B_heat_central_0_1)  c_e_equate_flows_heat_central_B(0_1)_  0.69
    flow(B_ch4_boiler_large_B_heat_central_0_1)  c_e_BusBlock_balance(B_heat_central_0_1)_  1
    flow(B_ch4_boiler_large_B_heat_central_0_2)  OBJ       1
    flow(B_ch4_boiler_large_B_heat_central_0_2)  c_e_equate_flows_heat_central_B(0_2)_  0.69
    flow(B_ch4_boiler_large_B_heat_central_0_2)  c_e_BusBlock_balance(B_heat_central_0_2)_  1
    flow(BB_ch4_boiler_large_BB_heat_central_0_0)  OBJ       1
    flow(BB_ch4_boiler_large_BB_heat_central_0_0)  c_e_equate_flows_heat_central_BB(0_0)_  0.69
    flow(BB_ch4_boiler_large_BB_heat_central_0_0)  c_e_BusBlock_balance(BB_heat_central_0_0)_  1
    flow(BB_ch4_boiler_large_BB_heat_central_0_1)  OBJ       1
    flow(BB_ch4_boiler_large_BB_heat_central_0_1)  c_e_equate_flows_heat_central_BB(0_1)_  0.69
    flow(BB_ch4_boiler_large_BB_heat_central_0_1)  c_e_BusBlock_balance(BB_heat_central_0_1)_  1
    flow(BB_ch4_boiler_large_BB_heat_central_0_2)  OBJ       1
    flow(BB_ch4_boiler_large_BB_heat_central_0_2)  c_e_equate_flows_heat_central_BB(0_2)_  0.69
    flow(BB_ch4_boiler_large_BB_heat_central_0_2)  c_e_BusBlock_balance(BB_heat_central_0_2)_  1
    flow(B_ch4_gt_B_electricity_0_0)  OBJ       4
    flow(B_ch4_gt_B_electricity_0_0)  c_e_BusBlock_balance(B_electricity_0_0)_  1
    flow(B_ch4_gt_B_electricity_0_1)  OBJ       4
    flow(B_ch4_gt_B_electricity_0_1)  c_e_BusBlock_balance(B_electricity_0_1)_  1
    flow(B_ch4_gt_B_electricity_0_2)  OBJ       4
    flow(B_ch4_gt_B_electricity_0_2)  c_e_BusBlock_balance(B_electricity_0_2)_  1
    flow(BB_ch4_gt_BB_electricity_0_0)  OBJ       4
    flow(BB_ch4_gt_BB_electricity_0_0)  c_e_BusBlock_balance(BB_electricity_0_0)_  1
    flow(BB_ch4_gt_BB_electricity_0_1)  OBJ       4
    flow(BB_ch4_gt_BB_electricity_0_1)  c_e_BusBlock_balance(BB_electricity_0_1)_  1
    flow(BB_ch4_gt_BB_electricity_0_2)  OBJ       4
    flow(BB_ch4_gt_BB_electricity_0_2)  c_e_BusBlock_balance(BB_electricity_0_2)_  1
    flow(B_ch4_import_B_ch4_0_0)  OBJ       46.6
    flow(B_ch4_import_B_ch4_0_0)  c_u_integral_limit_emission_factor_constraint_  191.15
    flow(B_ch4_import_B_ch4_0_0)  c_e_BusBlock_balance(B_ch4_0_0)_  1
    flow(B_ch4_import_B_ch4_0_1)  OBJ       46.6
    flow(B_ch4_import_B_ch4_0_1)  c_u_integral_limit_emission_factor_constraint_  191.15
    flow(B_ch4_import_B_ch4_0_1)  c_e_BusBlock_balance(B_ch4_0_1)_  1
    flow(B_ch4_import_B_ch4_0_2)  OBJ       46.6
    flow(B_ch4_import_B_ch4_0_2)  c_u_integral_limit_emission_factor_constraint_  191.15
    flow(B_ch4_import_B_ch4_0_2)  c_e_BusBlock_balance(B_ch4_0_2)_  1
    flow(BB_ch4_import_BB_ch4_0_0)  OBJ       46.6
    flow(BB_ch4_import_BB_ch4_0_0)  c_u_integral_limit_emission_factor_constraint_  191.15
    flow(BB_ch4_import_BB_ch4_0_0)  c_e_BusBlock_balance(BB_ch4_0_0)_  1
    flow(BB_ch4_import_BB_ch4_0_1)  OBJ       46.6
    flow(BB_ch4_import_BB_ch4_0_1)  c_u_integral_limit_emission_factor_constraint_  191.15
    flow(BB_ch4_import_BB_ch4_0_1)  c_e_BusBlock_balance(BB_ch4_0_1)_  1
    flow(BB_ch4_import_BB_ch4_0_2)  OBJ       46.6
    flow(BB_ch4_import_BB_ch4_0_2)  c_u_integral_limit_emission_factor_constraint_  191.15
    flow(BB_ch4_import_BB_ch4_0_2)  c_e_BusBlock_balance(BB_ch4_0_2)_  1
    flow(B_ch4_shortage_B_ch4_0_0)  OBJ       208.7
    flow(B_ch4_shortage_B_ch4_0_0)  c_e_BusBlock_balance(B_ch4_0_0)_  1
    flow(B_ch4_shortage_B_ch4_0_1)  OBJ       208.7
    flow(B_ch4_shortage_B_ch4_0_1)  c_e_BusBlock_balance(B_ch4_0_1)_  1
    flow(B_ch4_shortage_B_ch4_0_2)  OBJ       208.7
    flow(B_ch4_shortage_B_ch4_0_2)  c_e_BusBlock_balance(B_ch4_0_2)_  1
    flow(BB_ch4_shortage_BB_ch4_0_0)  OBJ       208.7
    flow(BB_ch4_shortage_BB_ch4_0_0)  c_e_BusBlock_balance(BB_ch4_0_0)_  1
    flow(BB_ch4_shortage_BB_ch4_0_1)  OBJ       208.7
    flow(BB_ch4_shortage_BB_ch4_0_1)  c_e_BusBlock_balance(BB_ch4_0_1)_  1
    flow(BB_ch4_shortage_BB_ch4_0_2)  OBJ       208.7
    flow(BB_ch4_shortage_BB_ch4_0_2)  c_e_BusBlock_balance(BB_ch4_0_2)_  1
    flow(B_electricity_heatpump_large_B_heat_central_0_0)  OBJ       1.7
    flow(B_electricity_heatpump_large_B_heat_central_0_0)  c_e_equate_flows_heat_central_B(0_0)_  -1
    flow(B_electricity_heatpump_large_B_heat_central_0_0)  c_e_BusBlock_balance(B_heat_central_0_0)_  1
    flow(B_electricity_heatpump_large_B_heat_central_0_1)  OBJ       1.7
    flow(B_electricity_heatpump_large_B_heat_central_0_1)  c_e_equate_flows_heat_central_B(0_1)_  -1
    flow(B_electricity_heatpump_large_B_heat_central_0_1)  c_e_BusBlock_balance(B_heat_central_0_1)_  1
    flow(B_electricity_heatpump_large_B_heat_central_0_2)  OBJ       1.7
    flow(B_electricity_heatpump_large_B_heat_central_0_2)  c_e_equate_flows_heat_central_B(0_2)_  -1
    flow(B_electricity_heatpump_large_B_heat_central_0_2)  c_e_BusBlock_balance(B_heat_central_0_2)_  1
    flow(BB_electricity_heatpump_large_BB_heat_central_0_0)  OBJ       1.7
    flow(BB_electricity_heatpump_large_BB_heat_central_0_0)  c_e_equate_flows_heat_central_BB(0_0)_  -1
    flow(BB_electricity_heatpump_large_BB_heat_central_0_0)  c_e_BusBlock_balance(BB_heat_central_0_0)_  1
    flow(BB_electricity_heatpump_large_BB_heat_central_0_1)  OBJ       1.7
    flow(BB_electricity_heatpump_large_BB_heat_central_0_1)  c_e_equate_flows_heat_central_BB(0_1)_  -1
    flow(BB_electricity_heatpump_large_BB_heat_central_0_1)  c_e_BusBlock_balance(BB_heat_central_0_1)_  1
    flow(BB_electricity_heatpump_large_BB_heat_central_0_2)  OBJ       1.7
    flow(BB_electricity_heatpump_large_BB_heat_central_0_2)  c_e_equate_flows_heat_central_BB(0_2)_  -1
    flow(BB_electricity_heatpump_large_BB_heat_central_0_2)  c_e_BusBlock_balance(BB_heat_central_0_2)_  1
    flow(B_electricity_heatpump_small_B_heat_decentral_0_0)  OBJ       2.7
    flow(B_electricity_heatpump_small_B_heat_decentral_0_0)  c_e_equate_flows_heat_decentral_B(0_0)_  -1
    flow(B_electricity_heatpump_small_B_heat_decentral_0_0)  c_e_BusBlock_balance(B_heat_decentral_0_0)_  1
    flow(B_electricity_heatpump_small_B_heat_decentral_0_1)  OBJ       2.7
    flow(B_electricity_heatpump_small_B_heat_decentral_0_1)  c_e_equate_flows_heat_decentral_B(0_1)_  -1
    flow(B_electricity_heatpump_small_B_heat_decentral_0_1)  c_e_BusBlock_balance(B_heat_decentral_0_1)_  1
    flow(B_electricity_heatpump_small_B_heat_decentral_0_2)  OBJ       2.7
    flow(B_electricity_heatpump_small_B_heat_decentral_0_2)  c_e_equate_flows_heat_decentral_B(0_2)_  -1
    flow(B_electricity_heatpump_small_B_heat_decentral_0_2)  c_e_BusBlock_balance(B_heat_decentral_0_2)_  1
    flow(BB_electricity_heatpump_small_BB_heat_decentral_0_0)  OBJ       2.7
    flow(BB_electricity_heatpump_small_BB_heat_decentral_0_0)  c_e_equate_flows_heat_decentral_BB(0_0)_  -1
    flow(BB_electricity_heatpump_small_BB_heat_decentral_0_0)  c_e_BusBlock_balance(BB_heat_decentral_0_0)_  1
    flow(BB_electricity_heatpump_small_BB_heat_decentral_0_1)  OBJ       2.7
    flow(BB_electricity_heatpump_small_BB_heat_decentral_0_1)  c_e_equate_flows_heat_decentral_BB(0_1)_  -1
    flow(BB_electricity_heatpump_small_BB_heat_decentral_0_1)  c_e_BusBlock_balance(BB_heat_decentral_0_1)_  1
    flow(BB_electricity_heatpump_small_BB_heat_decentral_0_2)  OBJ       2.7
    flow(BB_electricity_heatpump_small_BB_heat_decentral_0_2)  c_e_equate_flows_heat_decentral_BB(0_2)_  -1
    flow(BB_electricity_heatpump_small_BB_heat_decentral_0_2)  c_e_BusBlock_balance(BB_heat_decentral_0_2)_  1
    flow(B_electricity_import_B_electricity_0_0)  OBJ       82
    flow(B_electricity_import_B_electricity_0_0)  c_e_BusBlock_balance(B_electricity_0_0)_  1
    flow(B_electricity_import_B_electricity_0_1)  OBJ       82
    flow(B_electricity_import_B_electricity_0_1)  c_e_BusBlock_balance(B_electricity_0_1)_  1
    flow(B_electricity_import_B_electricity_0_2)  OBJ       82
    flow(B_electricity_import_B_electricity_0_2)  c_e_BusBlock_balance(B_electricity_0_2)_  1
    flow(BB_electricity_import_BB_electricity_0_0)  OBJ       82
    flow(BB_electricity_import_BB_electricity_0_0)  c_u_integral_limit_emission_factor_constraint_  0.02
    flow(BB_electricity_import_BB_electricity_0_0)  c_e_BusBlock_balance(BB_electricity_0_0)_  1
    flow(BB_electricity_import_BB_electricity_0_1)  OBJ       82
    flow(BB_electricity_import_BB_electricity_0_1)  c_u_integral_limit_emission_factor_constraint_  0.02
    flow(BB_electricity_import_BB_electricity_0_1)  c_e_BusBlock_balance(BB_electricity_0_1)_  1
    flow(BB_electricity_import_BB_electricity_0_2)  OBJ       82
    flow(BB_electricity_import_BB_electricity_0_2)  c_u_integral_limit_emission_factor_constraint_  0.02
    flow(BB_electricity_import_BB_electricity_0_2)  c_e_BusBlock_balance(BB_electricity_0_2)_  1
    flow(B_electricity_liion_battery_B_electricity_0_0)  OBJ       1.6
    flow(B_electricity_liion_battery_B_electricity_0_0)  c_e_BusBlock_balance(B_electricity_0_0)_  1
    flow(B_electricity_liion_battery_B_electricity_0_1)  OBJ       1.6
    flow(B_electricity_liion_battery_B_electricity_0_1)  c_e_BusBlock_balance(B_electricity_0_1)_  1
    flow(B_electricity_liion_battery_B_electricity_0_2)  OBJ       1.6
    flow(B_electricity_liion_battery_B_electricity_0_2)  c_e_BusBlock_balance(B_electricity_0_2)_  1
    flow(BB_electricity_liion_battery_BB_electricity_0_0)  OBJ       1.6
    flow(BB_electricity_liion_battery_BB_electricity_0_0)  c_e_BusBlock_balance(BB_electricity_0_0)_  1
    flow(BB_electricity_liion_battery_BB_electricity_0_1)  OBJ       1.6
    flow(BB_electricity_liion_battery_BB_electricity_0_1)  c_e_BusBlock_balance(BB_electricity_0_1)_  1
    flow(BB_electricity_liion_battery_BB_electricity_0_2)  OBJ       1.6
    flow(BB_electricity_liion_battery_BB_electricity_0_2)  c_e_BusBlock_balance(BB_electricity_0_2)_  1
    flow(B_electricity_pth_B_heat_central_0_0)  OBJ       1
    flow(B_electricity_pth_B_heat_central_0_0)  c_e_equate_flows_heat_central_B(0_0)_  -1
    flow(B_electricity_pth_B_heat_central_0_0)  c_e_BusBlock_balance(B_heat_central_0_0)_  1
    flow(B_electricity_pth_B_heat_central_0_1)  OBJ       1
    flow(B_electricity_pth_B_heat_central_0_1)  c_e_equate_flows_heat_central_B(0_1)_  -1
    flow(B_electricity_pth_B_heat_central_0_1)  c_e_BusBlock_balance(B_heat_central_0_1)_  1
    flow(B_electricity_pth_B_heat_central_0_2)  OBJ       1
    flow(B_electricity_pth_B_heat_central_0_2)  c_e_equate_flows_heat_central_B(0_2)_  -1
    flow(B_electricity_pth_B_heat_central_0_2)  c_e_BusBlock_balance(B_heat_central_0_2)_  1
    flow(BB_electricity_pth_BB_heat_central_0_0)  OBJ       1
    flow(BB_electricity_pth_BB_heat_central_0_0)  c_e_equate_flows_heat_central_BB(0_0)_  -1
    flow(BB_electricity_pth_BB_heat_central_0_0)  c_e_BusBlock_balance(BB_heat_central_0_0)_  1
    flow(BB_electricity_pth_BB_heat_central_0_1)  OBJ       1
    flow(BB_electricity_pth_BB_heat_central_0_1)  c_e_equate_flows_heat_central_BB(0_1)_  -1
    flow(BB_electricity_pth_BB_heat_central_0_1)  c_e_BusBlock_balance(BB_heat_central_0_1)_  1
    flow(BB_electricity_pth_BB_heat_central_0_2)  OBJ       1
    flow(BB_electricity_pth_BB_heat_central_0_2)  c_e_equate_flows_heat_central_BB(0_2)_  -1
    flow(BB_electricity_pth_BB_heat_central_0_2)  c_e_BusBlock_balance(BB_heat_central_0_2)_  1
    flow(B_electricity_shortage_B_electricity_0_0)  OBJ       1000000000
    flow(B_electricity_shortage_B_electricity_0_0)  c_e_BusBlock_balance(B_electricity_0_0)_  1
    flow(B_electricity_shortage_B_electricity_0_1)  OBJ       1000000000
    flow(B_electricity_shortage_B_electricity_0_1)  c_e_BusBlock_balance(B_electricity_0_1)_  1
    flow(B_electricity_shortage_B_electricity_0_2)  OBJ       1000000000
    flow(B_electricity_shortage_B_electricity_0_2)  c_e_BusBlock_balance(B_electricity_0_2)_  1
    flow(BB_electricity_shortage_BB_electricity_0_0)  OBJ       1000000000
    flow(BB_electricity_shortage_BB_electricity_0_0)  c_e_BusBlock_balance(BB_electricity_0_0)_  1
    flow(BB_electricity_shortage_BB_electricity_0_1)  OBJ       1000000000
    flow(BB_electricity_shortage_BB_electricity_0_1)  c_e_BusBlock_balance(BB_electricity_0_1)_  1
    flow(BB_electricity_shortage_BB_electricity_0_2)  OBJ       1000000000
    flow(BB_electricity_shortage_BB_electricity_0_2)  c_e_BusBlock_balance(BB_electricity_0_2)_  1
    flow(B_h2_gt_B_electricity_0_0)  OBJ       4
    flow(B_h2_gt_B_electricity_0_0)  c_e_BusBlock_balance(B_electricity_0_0)_  1
    flow(B_h2_gt_B_electricity_0_1)  OBJ       4
    flow(B_h2_gt_B_electricity_0_1)  c_e_BusBlock_balance(B_electricity_0_1)_  1
    flow(B_h2_gt_B_electricity_0_2)  OBJ       4
    flow(B_h2_gt_B_electricity_0_2)  c_e_BusBlock_balance(B_electricity_0_2)_  1
    flow(BB_h2_gt_BB_electricity_0_0)  OBJ       4
    flow(BB_h2_gt_BB_electricity_0_0)  c_e_BusBlock_balance(BB_electricity_0_0)_  1
    flow(BB_h2_gt_BB_electricity_0_1)  OBJ       4
    flow(BB_h2_gt_BB_electricity_0_1)  c_e_BusBlock_balance(BB_electricity_0_1)_  1
    flow(BB_h2_gt_BB_electricity_0_2)  OBJ       4
    flow(BB_h2_gt_BB_electricity_0_2)  c_e_BusBlock_balance(BB_electricity_0_2)_  1
    flow(B_h2_import_B_h2_0_0)  OBJ       88.8
    flow(B_h2_import_B_h2_0_0)  c_u_integral_limit_emission_factor_constraint_  79.81
    flow(B_h2_import_B_h2_0_0)  c_e_BusBlock_balance(B_h2_0_0)_  1
    flow(B_h2_import_B_h2_0_1)  OBJ       88.8
    flow(B_h2_import_B_h2_0_1)  c_u_integral_limit_emission_factor_constraint_  79.81
    flow(B_h2_import_B_h2_0_1)  c_e_BusBlock_balance(B_h2_0_1)_  1
    flow(B_h2_import_B_h2_0_2)  OBJ       88.8
    flow(B_h2_import_B_h2_0_2)  c_u_integral_limit_emission_factor_constraint_  79.81
    flow(B_h2_import_B_h2_0_2)  c_e_BusBlock_balance(B_h2_0_2)_  1
    flow(BB_h2_import_BB_h2_0_0)  OBJ       88.8
    flow(BB_h2_import_BB_h2_0_0)  c_u_integral_limit_emission_factor_constraint_  79.81
    flow(BB_h2_import_BB_h2_0_0)  c_e_BusBlock_balance(BB_h2_0_0)_  1
    flow(BB_h2_import_BB_h2_0_1)  OBJ       88.8
    flow(BB_h2_import_BB_h2_0_1)  c_u_integral_limit_emission_factor_constraint_  79.81
    flow(BB_h2_import_BB_h2_0_1)  c_e_BusBlock_balance(BB_h2_0_1)_  1
    flow(BB_h2_import_BB_h2_0_2)  OBJ       88.8
    flow(BB_h2_import_BB_h2_0_2)  c_u_integral_limit_emission_factor_constraint_  79.81
    flow(BB_h2_import_BB_h2_0_2)  c_e_BusBlock_balance(BB_h2_0_2)_  1
    flow(B_h2_shortage_B_h2_0_0)  OBJ       1000000000
    flow(B_h2_shortage_B_h2_0_0)  c_e_BusBlock_balance(B_h2_0_0)_  1
    flow(B_h2_shortage_B_h2_0_1)  OBJ       1000000000
    flow(B_h2_shortage_B_h2_0_1)  c_e_BusBlock_balance(B_h2_0_1)_  1
    flow(B_h2_shortage_B_h2_0_2)  OBJ       1000000000
    flow(B_h2_shortage_B_h2_0_2)  c_e_BusBlock_balance(B_h2_0_2)_  1
    flow(BB_h2_shortage_BB_h2_0_0)  OBJ       1000000000
    flow(BB_h2_shortage_BB_h2_0_0)  c_e_BusBlock_balance(BB_h2_0_0)_  1
    flow(BB_h2_shortage_BB_h2_0_1)  OBJ       1000000000
    flow(BB_h2_shortage_BB_h2_0_1)  c_e_BusBlock_balance(BB_h2_0_1)_  1
    flow(BB_h2_shortage_BB_h2_0_2)  OBJ       1000000000
    flow(BB_h2_shortage_BB_h2_0_2)  c_e_BusBlock_balance(BB_h2_0_2)_  1
    flow(B_heat_central_shortage_B_heat_central_0_0)  OBJ       1000000000
    flow(B_heat_central_shortage_B_heat_central_0_0)  c_e_BusBlock_balance(B_heat_central_0_0)_  1
    flow(B_heat_central_shortage_B_heat_central_0_1)  OBJ       1000000000
    flow(B_heat_central_shortage_B_heat_central_0_1)  c_e_BusBlock_balance(B_heat_central_0_1)_  1
    flow(B_heat_central_shortage_B_heat_central_0_2)  OBJ       1000000000
    flow(B_heat_central_shortage_B_heat_central_0_2)  c_e_BusBlock_balance(B_heat_central_0_2)_  1
    flow(BB_heat_central_shortage_BB_heat_central_0_0)  OBJ       1000000000
    flow(BB_heat_central_shortage_BB_heat_central_0_0)  c_e_BusBlock_balance(BB_heat_central_0_0)_  1
    flow(BB_heat_central_shortage_BB_heat_central_0_1)  OBJ       1000000000
    flow(BB_heat_central_shortage_BB_heat_central_0_1)  c_e_BusBlock_balance(BB_heat_central_0_1)_  1
    flow(BB_heat_central_shortage_BB_heat_central_0_2)  OBJ       1000000000
    flow(BB_heat_central_shortage_BB_heat_central_0_2)  c_e_BusBlock_balance(BB_heat_central_0_2)_  1
    flow(B_heat_decentral_shortage_B_heat_decentral_0_0)  OBJ       1000000000
    flow(B_heat_decentral_shortage_B_heat_decentral_0_0)  c_e_BusBlock_balance(B_heat_decentral_0_0)_  1
    flow(B_heat_decentral_shortage_B_heat_decentral_0_1)  OBJ       1000000000
    flow(B_heat_decentral_shortage_B_heat_decentral_0_1)  c_e_BusBlock_balance(B_heat_decentral_0_1)_  1
    flow(B_heat_decentral_shortage_B_heat_decentral_0_2)  OBJ       1000000000
    flow(B_heat_decentral_shortage_B_heat_decentral_0_2)  c_e_BusBlock_balance(B_heat_decentral_0_2)_  1
    flow(BB_heat_decentral_shortage_BB_heat_decentral_0_0)  OBJ       1000000000
    flow(BB_heat_decentral_shortage_BB_heat_decentral_0_0)  c_e_BusBlock_balance(BB_heat_decentral_0_0)_  1
    flow(BB_heat_decentral_shortage_BB_heat_decentral_0_1)  OBJ       1000000000
    flow(BB_heat_decentral_shortage_BB_heat_decentral_0_1)  c_e_BusBlock_balance(BB_heat_decentral_0_1)_  1
    flow(BB_heat_decentral_shortage_BB_heat_decentral_0_2)  OBJ       1000000000
    flow(BB_heat_decentral_shortage_BB_heat_decentral_0_2)  c_e_BusBlock_balance(BB_heat_decentral_0_2)_  1
    flow(B_heat_decentral_storage_B_heat_decentral_0_0)  OBJ       1.2
    flow(B_heat_decentral_storage_B_heat_decentral_0_0)  c_e_BusBlock_balance(B_heat_decentral_0_0)_  1
    flow(B_heat_decentral_storage_B_heat_decentral_0_1)  OBJ       1.2
    flow(B_heat_decentral_storage_B_heat_decentral_0_1)  c_e_BusBlock_balance(B_heat_decentral_0_1)_  1
    flow(B_heat_decentral_storage_B_heat_decentral_0_2)  OBJ       1.2
    flow(B_heat_decentral_storage_B_heat_decentral_0_2)  c_e_BusBlock_balance(B_heat_decentral_0_2)_  1
    flow(BB_heat_decentral_storage_BB_heat_decentral_0_0)  OBJ       1.2
    flow(BB_heat_decentral_storage_BB_heat_decentral_0_0)  c_e_BusBlock_balance(BB_heat_decentral_0_0)_  1
    flow(BB_heat_decentral_storage_BB_heat_decentral_0_1)  OBJ       1.2
    flow(BB_heat_decentral_storage_BB_heat_decentral_0_1)  c_e_BusBlock_balance(BB_heat_decentral_0_1)_  1
    flow(BB_heat_decentral_storage_BB_heat_decentral_0_2)  OBJ       1.2
    flow(BB_heat_decentral_storage_BB_heat_decentral_0_2)  c_e_BusBlock_balance(BB_heat_decentral_0_2)_  1
    flow(B_wind_onshore_B_electricity_0_0)  OBJ       1.22
    flow(B_wind_onshore_B_electricity_0_0)  c_e_BusBlock_balance(B_electricity_0_0)_  1
    flow(B_wind_onshore_B_electricity_0_1)  OBJ       1.22
    flow(B_wind_onshore_B_electricity_0_1)  c_e_BusBlock_balance(B_electricity_0_1)_  1
    flow(B_wind_onshore_B_electricity_0_2)  OBJ       1.22
    flow(B_wind_onshore_B_electricity_0_2)  c_e_BusBlock_balance(B_electricity_0_2)_  1
    flow(BB_wind_onshore_BB_electricity_0_0)  OBJ       1.22
    flow(BB_wind_onshore_BB_electricity_0_0)  c_e_BusBlock_balance(BB_electricity_0_0)_  1
    flow(BB_wind_onshore_BB_electricity_0_1)  OBJ       1.22
    flow(BB_wind_onshore_BB_electricity_0_1)  c_e_BusBlock_balance(BB_electricity_0_1)_  1
    flow(BB_wind_onshore_BB_electricity_0_2)  OBJ       1.22
    flow(BB_wind_onshore_BB_electricity_0_2)  c_e_BusBlock_balance(BB_electricity_0_2)_  1
    GenericInvestmentStorageBlock_invest(BB_heat_decentral_storage_0)  OBJ       4.0376340644801137e+04
    GenericInvestmentStorageBlock_invest(BB_heat_decentral_storage_0)  c_u_GenericInvestmentStorageBlock_init_content_limit(BB_heat_decentral_storage)_  -1
    GenericInvestmentStorageBlock_invest(B_h2_cavern_0)  OBJ       7.2969600045258744e+01
    GenericInvestmentStorageBlock_invest(B_h2_cavern_0)  c_u_GenericInvestmentStorageBlock_init_content_limit(B_h2_cavern)_  -1
    GenericInvestmentStorageBlock_invest(B_heat_decentral_storage_0)  OBJ       4.0376340644801137e+04
    GenericInvestmentStorageBlock_invest(B_heat_decentral_storage_0)  c_u_GenericInvestmentStorageBlock_init_content_limit(B_heat_decentral_storage)_  -1
    GenericInvestmentStorageBlock_invest(BB_h2_cavern_0)  OBJ       7.2969600045258744e+01
    GenericInvestmentStorageBlock_invest(BB_h2_cavern_0)  c_u_GenericInvestmentStorageBlock_init_content_limit(BB_h2_cavern)_  -1
    GenericInvestmentStorageBlock_invest(B_electricity_liion_battery_0)  OBJ       1.6021675279083636e+04
    GenericInvestmentStorageBlock_invest(B_electricity_liion_battery_0)  c_u_GenericInvestmentStorageBlock_init_content_limit(B_electricity_liion_battery)_  -1
    GenericInvestmentStorageBlock_invest(BB_heat_central_storage_0)  OBJ       1.6017046797326662e+02
    GenericInvestmentStorageBlock_invest(BB_heat_central_storage_0)  c_u_GenericInvestmentStorageBlock_init_content_limit(BB_heat_central_storage)_  -1
    GenericInvestmentStorageBlock_invest(BB_electricity_liion_battery_0)  OBJ       1.6021675279083636e+04
    GenericInvestmentStorageBlock_invest(BB_electricity_liion_battery_0)  c_u_GenericInvestmentStorageBlock_init_content_limit(BB_electricity_liion_battery)_  -1
    GenericInvestmentStorageBlock_invest(B_heat_central_storage_0)  OBJ       1.6017046797326662e+02
    GenericInvestmentStorageBlock_invest(B_heat_central_storage_0)  c_u_GenericInvestmentStorageBlock_init_content_limit(B_heat_central_storage)_  -1
    GenericInvestmentStorageBlock_init_content(BB_heat_decentral_storage)  c_u_GenericInvestmentStorageBlock_init_content_limit(BB_heat_decentral_storage)_  1
    GenericInvestmentStorageBlock_init_content(B_h2_cavern)  c_u_GenericInvestmentStorageBlock_init_content_limit(B_h2_cavern)_  1
    GenericInvestmentStorageBlock_init_content(B_heat_decentral_storage)  c_u_GenericInvestmentStorageBlock_init_content_limit(B_heat_decentral_storage)_  1
    GenericInvestmentStorageBlock_init_content(BB_h2_cavern)  c_u_GenericInvestmentStorageBlock_init_content_limit(BB_h2_cavern)_  1
    GenericInvestmentStorageBlock_init_content(B_electricity_liion_battery)  c_u_GenericInvestmentStorageBlock_init_content_limit(B_electricity_liion_battery)_  1
    GenericInvestmentStorageBlock_init_content(BB_heat_central_storage)  c_u_GenericInvestmentStorageBlock_init_content_limit(BB_heat_central_storage)_  1
    GenericInvestmentStorageBlock_init_content(BB_electricity_liion_battery)  c_u_GenericInvestmentStorageBlock_init_content_limit(BB_electricity_liion_battery)_  1
    GenericInvestmentStorageBlock_init_content(B_heat_central_storage)  c_u_GenericInvestmentStorageBlock_init_content_limit(B_heat_central_storage)_  1
    InvestmentFlowBlock_invest(BB_heat_decentral_storage_BB_heat_decentral_0)  c_e_GenericInvestmentStorageBlock_power_coupled(BB_heat_decentral_storage_0)_  1
    InvestmentFlowBlock_invest(BB_heat_decentral_BB_heat_decentral_storage_0)  c_e_GenericInvestmentStorageBlock_power_coupled(BB_heat_decentral_storage_0)_  -1
    InvestmentFlowBlock_invest(B_h2_cavern_B_h2_0)  c_e_GenericInvestmentStorageBlock_power_coupled(B_h2_cavern_0)_  1
    InvestmentFlowBlock_invest(B_heat_decentral_B_heat_decentral_storage_0)  c_e_GenericInvestmentStorageBlock_power_coupled(B_heat_decentral_storage_0)_  -1
    InvestmentFlowBlock_invest(B_heat_decentral_storage_B_heat_decentral_0)  c_e_GenericInvestmentStorageBlock_power_coupled(B_heat_decentral_storage_0)_  1
    InvestmentFlowBlock_invest(BB_h2_cavern_BB_h2_0)  c_e_GenericInvestmentStorageBlock_power_coupled(BB_h2_cavern_0)_  1
    InvestmentFlowBlock_invest(B_electricity_liion_battery_B_electricity_0)  c_e_GenericInvestmentStorageBlock_power_coupled(B_electricity_liion_battery_0)_  1
    InvestmentFlowBlock_invest(BB_heat_central_BB_heat_central_storage_0)  c_e_GenericInvestmentStorageBlock_power_coupled(BB_heat_central_storage_0)_  -1
    InvestmentFlowBlock_invest(BB_heat_central_storage_BB_heat_central_0)  c_e_GenericInvestmentStorageBlock_power_coupled(BB_heat_central_storage_0)_  1
    InvestmentFlowBlock_invest(BB_electricity_liion_battery_BB_electricity_0)  c_e_GenericInvestmentStorageBlock_power_coupled(BB_electricity_liion_battery_0)_  1
    InvestmentFlowBlock_invest(B_heat_central_storage_B_heat_central_0)  c_e_GenericInvestmentStorageBlock_power_coupled(B_heat_central_storage_0)_  1
    InvestmentFlowBlock_invest(B_heat_central_B_heat_central_storage_0)  c_e_GenericInvestmentStorageBlock_power_coupled(B_heat_central_storage_0)_  -1
    flow(B_ch4_boiler_small_B_heat_decentral_0_0)  c_e_equate_flows_heat_decentral_B(0_0)_  1.3
    flow(B_ch4_boiler_small_B_heat_decentral_0_0)  c_e_BusBlock_balance(B_heat_decentral_0_0)_  1
    flow(B_ch4_boiler_small_B_heat_decentral_0_1)  c_e_equate_flows_heat_decentral_B(0_1)_  1.3
    flow(B_ch4_boiler_small_B_heat_decentral_0_1)  c_e_BusBlock_balance(B_heat_decentral_0_1)_  1
    flow(B_ch4_boiler_small_B_heat_decentral_0_2)  c_e_equate_flows_heat_decentral_B(0_2)_  1.3
    flow(B_ch4_boiler_small_B_heat_decentral_0_2)  c_e_BusBlock_balance(B_heat_decentral_0_2)_  1
    flow(BB_ch4_boiler_small_BB_heat_decentral_0_0)  c_e_equate_flows_heat_decentral_BB(0_0)_  4.9
    flow(BB_ch4_boiler_small_BB_heat_decentral_0_0)  c_e_BusBlock_balance(BB_heat_decentral_0_0)_  1
    flow(BB_ch4_boiler_small_BB_heat_decentral_0_1)  c_e_equate_flows_heat_decentral_BB(0_1)_  4.9
    flow(BB_ch4_boiler_small_BB_heat_decentral_0_1)  c_e_BusBlock_balance(BB_heat_decentral_0_1)_  1
    flow(BB_ch4_boiler_small_BB_heat_decentral_0_2)  c_e_equate_flows_heat_decentral_BB(0_2)_  4.9
    flow(BB_ch4_boiler_small_BB_heat_decentral_0_2)  c_e_BusBlock_balance(BB_heat_decentral_0_2)_  1
    flow(B_ch4_B_ch4_boiler_large_0_0)  c_e_BusBlock_balance(B_ch4_0_0)_  -1
    flow(B_ch4_B_ch4_boiler_small_0_0)  c_e_BusBlock_balance(B_ch4_0_0)_  -1
    flow(B_ch4_B_ch4_bpchp_0_0)  c_e_BusBlock_balance(B_ch4_0_0)_  -1
    flow(B_ch4_B_ch4_excess_0_0)  c_e_BusBlock_balance(B_ch4_0_0)_  -1
    flow(B_ch4_B_ch4_export_0_0)  c_e_BusBlock_balance(B_ch4_0_0)_  -1
    flow(B_ch4_B_ch4_gt_0_0)  c_e_BusBlock_balance(B_ch4_0_0)_  -1
    flow(B_ch4_B_ch4_boiler_large_0_1)  c_e_BusBlock_balance(B_ch4_0_1)_  -1
    flow(B_ch4_B_ch4_boiler_small_0_1)  c_e_BusBlock_balance(B_ch4_0_1)_  -1
    flow(B_ch4_B_ch4_bpchp_0_1)  c_e_BusBlock_balance(B_ch4_0_1)_  -1
    flow(B_ch4_B_ch4_excess_0_1)  c_e_BusBlock_balance(B_ch4_0_1)_  -1
    flow(B_ch4_B_ch4_export_0_1)  c_e_BusBlock_balance(B_ch4_0_1)_  -1
    flow(B_ch4_B_ch4_gt_0_1)  c_e_BusBlock_balance(B_ch4_0_1)_  -1
    flow(B_ch4_B_ch4_boiler_large_0_2)  c_e_BusBlock_balance(B_ch4_0_2)_  -1
    flow(B_ch4_B_ch4_boiler_small_0_2)  c_e_BusBlock_balance(B_ch4_0_2)_  -1
    flow(B_ch4_B_ch4_bpchp_0_2)  c_e_BusBlock_balance(B_ch4_0_2)_  -1
    flow(B_ch4_B_ch4_excess_0_2)  c_e_BusBlock_balance(B_ch4_0_2)_  -1
    flow(B_ch4_B_ch4_export_0_2)  c_e_BusBlock_balance(B_ch4_0_2)_  -1
    flow(B_ch4_B_ch4_gt_0_2)  c_e_BusBlock_balance(B_ch4_0_2)_  -1
    flow(BB_heat_decentral_BB_heat_decentral_storage_0_0)  c_e_BusBlock_balance(BB_heat_decentral_0_0)_  -1
    flow(BB_heat_decentral_BB_heat_decentral_excess_0_0)  c_e_BusBlock_balance(BB_heat_decentral_0_0)_  -1
    flow(BB_heat_decentral_BB_heat_decentral_storage_0_1)  c_e_BusBlock_balance(BB_heat_decentral_0_1)_  -1
    flow(BB_heat_decentral_BB_heat_decentral_excess_0_1)  c_e_BusBlock_balance(BB_heat_decentral_0_1)_  -1
    flow(BB_heat_decentral_BB_heat_decentral_storage_0_2)  c_e_BusBlock_balance(BB_heat_decentral_0_2)_  -1
    flow(BB_heat_decentral_BB_heat_decentral_excess_0_2)  c_e_BusBlock_balance(BB_heat_decentral_0_2)_  -1
    flow(B_heat_central_B_heat_central_storage_0_0)  c_e_BusBlock_balance(B_heat_central_0_0)_  -1
    flow(B_h2_bpchp_B_heat_central_0_0)  c_e_BusBlock_balance(B_heat_central_0_0)_  1
    flow(B_ch4_bpchp_B_heat_central_0_0)  c_e_BusBlock_balance(B_heat_central_0_0)_  1
    flow(B_heat_central_storage_B_heat_central_0_0)  c_e_BusBlock_balance(B_heat_central_0_0)_  1
    flow(B_heat_central_B_heat_central_excess_0_0)  c_e_BusBlock_balance(B_heat_central_0_0)_  -1
    flow(B_heat_central_B_heat_central_storage_0_1)  c_e_BusBlock_balance(B_heat_central_0_1)_  -1
    flow(B_h2_bpchp_B_heat_central_0_1)  c_e_BusBlock_balance(B_heat_central_0_1)_  1
    flow(B_ch4_bpchp_B_heat_central_0_1)  c_e_BusBlock_balance(B_heat_central_0_1)_  1
    flow(B_heat_central_storage_B_heat_central_0_1)  c_e_BusBlock_balance(B_heat_central_0_1)_  1
    flow(B_heat_central_B_heat_central_excess_0_1)  c_e_BusBlock_balance(B_heat_central_0_1)_  -1
    flow(B_heat_central_B_heat_central_storage_0_2)  c_e_BusBlock_balance(B_heat_central_0_2)_  -1
    flow(B_h2_bpchp_B_heat_central_0_2)  c_e_BusBlock_balance(B_heat_central_0_2)_  1
    flow(B_ch4_bpchp_B_heat_central_0_2)  c_e_BusBlock_balance(B_heat_central_0_2)_  1
    flow(B_heat_central_storage_B_heat_central_0_2)  c_e_BusBlock_balance(B_heat_central_0_2)_  1
    flow(B_heat_central_B_heat_central_excess_0_2)  c_e_BusBlock_balance(B_heat_central_0_2)_  -1
    flow(BB_ch4_BB_ch4_boiler_large_0_0)  c_e_BusBlock_balance(BB_ch4_0_0)_  -1
    flow(BB_ch4_BB_ch4_boiler_small_0_0)  c_e_BusBlock_balance(BB_ch4_0_0)_  -1
    flow(BB_ch4_BB_ch4_bpchp_0_0)  c_e_BusBlock_balance(BB_ch4_0_0)_  -1
    flow(BB_ch4_BB_ch4_excess_0_0)  c_e_BusBlock_balance(BB_ch4_0_0)_  -1
    flow(BB_ch4_BB_ch4_export_0_0)  c_e_BusBlock_balance(BB_ch4_0_0)_  -1
    flow(BB_ch4_BB_ch4_gt_0_0)  c_e_BusBlock_balance(BB_ch4_0_0)_  -1
    flow(BB_ch4_BB_ch4_boiler_large_0_1)  c_e_BusBlock_balance(BB_ch4_0_1)_  -1
    flow(BB_ch4_BB_ch4_boiler_small_0_1)  c_e_BusBlock_balance(BB_ch4_0_1)_  -1
    flow(BB_ch4_BB_ch4_bpchp_0_1)  c_e_BusBlock_balance(BB_ch4_0_1)_  -1
    flow(BB_ch4_BB_ch4_excess_0_1)  c_e_BusBlock_balance(BB_ch4_0_1)_  -1
    flow(BB_ch4_BB_ch4_export_0_1)  c_e_BusBlock_balance(BB_ch4_0_1)_  -1
    flow(BB_ch4_BB_ch4_gt_0_1)  c_e_BusBlock_balance(BB_ch4_0_1)_  -1
    flow(BB_ch4_BB_ch4_boiler_large_0_2)  c_e_BusBlock_balance(BB_ch4_0_2)_  -1
    flow(BB_ch4_BB_ch4_boiler_small_0_2)  c_e_BusBlock_balance(BB_ch4_0_2)_  -1
    flow(BB_ch4_BB_ch4_bpchp_0_2)  c_e_BusBlock_balance(BB_ch4_0_2)_  -1
    flow(BB_ch4_BB_ch4_excess_0_2)  c_e_BusBlock_balance(BB_ch4_0_2)_  -1
    flow(BB_ch4_BB_ch4_export_0_2)  c_e_BusBlock_balance(BB_ch4_0_2)_  -1
    flow(BB_ch4_BB_ch4_gt_0_2)  c_e_BusBlock_balance(BB_ch4_0_2)_  -1
    flow(B_h2_B_h2_cavern_0_0)  c_e_BusBlock_balance(B_h2_0_0)_  -1
    flow(B_electricity_electrolyzer_B_h2_0_0)  c_e_BusBlock_balance(B_h2_0_0)_  1
    flow(B_h2_cavern_B_h2_0_0)  c_e_BusBlock_balance(B_h2_0_0)_  1
    flow(B_h2_B_h2_bpchp_0_0)  c_e_BusBlock_balance(B_h2_0_0)_  -1
    flow(B_h2_B_h2_excess_0_0)  c_e_BusBlock_balance(B_h2_0_0)_  -1
    flow(B_h2_B_h2_export_0_0)  c_e_BusBlock_balance(B_h2_0_0)_  -1
    flow(B_h2_B_h2_gt_0_0)  c_e_BusBlock_balance(B_h2_0_0)_  -1
    flow(B_h2_B_h2_cavern_0_1)  c_e_BusBlock_balance(B_h2_0_1)_  -1
    flow(B_electricity_electrolyzer_B_h2_0_1)  c_e_BusBlock_balance(B_h2_0_1)_  1
    flow(B_h2_cavern_B_h2_0_1)  c_e_BusBlock_balance(B_h2_0_1)_  1
    flow(B_h2_B_h2_bpchp_0_1)  c_e_BusBlock_balance(B_h2_0_1)_  -1
    flow(B_h2_B_h2_excess_0_1)  c_e_BusBlock_balance(B_h2_0_1)_  -1
    flow(B_h2_B_h2_export_0_1)  c_e_BusBlock_balance(B_h2_0_1)_  -1
    flow(B_h2_B_h2_gt_0_1)  c_e_BusBlock_balance(B_h2_0_1)_  -1
    flow(B_h2_B_h2_cavern_0_2)  c_e_BusBlock_balance(B_h2_0_2)_  -1
    flow(B_electricity_electrolyzer_B_h2_0_2)  c_e_BusBlock_balance(B_h2_0_2)_  1
    flow(B_h2_cavern_B_h2_0_2)  c_e_BusBlock_balance(B_h2_0_2)_  1
    flow(B_h2_B_h2_bpchp_0_2)  c_e_BusBlock_balance(B_h2_0_2)_  -1
    flow(B_h2_B_h2_excess_0_2)  c_e_BusBlock_balance(B_h2_0_2)_  -1
    flow(B_h2_B_h2_export_0_2)  c_e_BusBlock_balance(B_h2_0_2)_  -1
    flow(B_h2_B_h2_gt_0_2)  c_e_BusBlock_balance(B_h2_0_2)_  -1
    flow(B_heat_decentral_B_heat_decentral_storage_0_0)  c_e_BusBlock_balance(B_heat_decentral_0_0)_  -1
    flow(B_heat_decentral_B_heat_decentral_excess_0_0)  c_e_BusBlock_balance(B_heat_decentral_0_0)_  -1
    flow(B_heat_decentral_B_heat_decentral_storage_0_1)  c_e_BusBlock_balance(B_heat_decentral_0_1)_  -1
    flow(B_heat_decentral_B_heat_decentral_excess_0_1)  c_e_BusBlock_balance(B_heat_decentral_0_1)_  -1
    flow(B_heat_decentral_B_heat_decentral_storage_0_2)  c_e_BusBlock_balance(B_heat_decentral_0_2)_  -1
    flow(B_heat_decentral_B_heat_decentral_excess_0_2)  c_e_BusBlock_balance(B_heat_decentral_0_2)_  -1
    flow(BB_heat_central_BB_heat_central_storage_0_0)  c_e_BusBlock_balance(BB_heat_central_0_0)_  -1
    flow(BB_h2_bpchp_BB_heat_central_0_0)  c_e_BusBlock_balance(BB_heat_central_0_0)_  1
    flow(BB_heat_central_storage_BB_heat_central_0_0)  c_e_BusBlock_balance(BB_heat_central_0_0)_  1
    flow(BB_ch4_bpchp_BB_heat_central_0_0)  c_e_BusBlock_balance(BB_heat_central_0_0)_  1
    flow(BB_heat_central_BB_heat_central_excess_0_0)  c_e_BusBlock_balance(BB_heat_central_0_0)_  -1
    flow(BB_heat_central_BB_heat_central_storage_0_1)  c_e_BusBlock_balance(BB_heat_central_0_1)_  -1
    flow(BB_h2_bpchp_BB_heat_central_0_1)  c_e_BusBlock_balance(BB_heat_central_0_1)_  1
    flow(BB_heat_central_storage_BB_heat_central_0_1)  c_e_BusBlock_balance(BB_heat_central_0_1)_  1
    flow(BB_ch4_bpchp_BB_heat_central_0_1)  c_e_BusBlock_balance(BB_heat_central_0_1)_  1
    flow(BB_heat_central_BB_heat_central_excess_0_1)  c_e_BusBlock_balance(BB_heat_central_0_1)_  -1
    flow(BB_heat_central_BB_heat_central_storage_0_2)  c_e_BusBlock_balance(BB_heat_central_0_2)_  -1
    flow(BB_h2_bpchp_BB_heat_central_0_2)  c_e_BusBlock_balance(BB_heat_central_0_2)_  1
    flow(BB_heat_central_storage_BB_heat_central_0_2)  c_e_BusBlock_balance(BB_heat_central_0_2)_  1
    flow(BB_ch4_bpchp_BB_heat_central_0_2)  c_e_BusBlock_balance(BB_heat_central_0_2)_  1
    flow(BB_heat_central_BB_heat_central_excess_0_2)  c_e_BusBlock_balance(BB_heat_central_0_2)_  -1
    flow(BB_h2_BB_h2_cavern_0_0)  c_e_BusBlock_balance(BB_h2_0_0)_  -1
    flow(BB_electricity_electrolyzer_BB_h2_0_0)  c_e_BusBlock_balance(BB_h2_0_0)_  1
    flow(BB_h2_cavern_BB_h2_0_0)  c_e_BusBlock_balance(BB_h2_0_0)_  1
    flow(BB_h2_BB_h2_bpchp_0_0)  c_e_BusBlock_balance(BB_h2_0_0)_  -1
    flow(BB_h2_BB_h2_excess_0_0)  c_e_BusBlock_balance(BB_h2_0_0)_  -1
    flow(BB_h2_BB_h2_export_0_0)  c_e_BusBlock_balance(BB_h2_0_0)_  -1
    flow(BB_h2_BB_h2_gt_0_0)  c_e_BusBlock_balance(BB_h2_0_0)_  -1
    flow(BB_h2_BB_h2_cavern_0_1)  c_e_BusBlock_balance(BB_h2_0_1)_  -1
    flow(BB_electricity_electrolyzer_BB_h2_0_1)  c_e_BusBlock_balance(BB_h2_0_1)_  1
    flow(BB_h2_cavern_BB_h2_0_1)  c_e_BusBlock_balance(BB_h2_0_1)_  1
    flow(BB_h2_BB_h2_bpchp_0_1)  c_e_BusBlock_balance(BB_h2_0_1)_  -1
    flow(BB_h2_BB_h2_excess_0_1)  c_e_BusBlock_balance(BB_h2_0_1)_  -1
    flow(BB_h2_BB_h2_export_0_1)  c_e_BusBlock_balance(BB_h2_0_1)_  -1
    flow(BB_h2_BB_h2_gt_0_1)  c_e_BusBlock_balance(BB_h2_0_1)_  -1
    flow(BB_h2_BB_h2_cavern_0_2)  c_e_BusBlock_balance(BB_h2_0_2)_  -1
    flow(BB_electricity_electrolyzer_BB_h2_0_2)  c_e_BusBlock_balance(BB_h2_0_2)_  1
    flow(BB_h2_cavern_BB_h2_0_2)  c_e_BusBlock_balance(BB_h2_0_2)_  1
    flow(BB_h2_BB_h2_bpchp_0_2)  c_e_BusBlock_balance(BB_h2_0_2)_  -1
    flow(BB_h2_BB_h2_excess_0_2)  c_e_BusBlock_balance(BB_h2_0_2)_  -1
    flow(BB_h2_BB_h2_export_0_2)  c_e_BusBlock_balance(BB_h2_0_2)_  -1
    flow(BB_h2_BB_h2_gt_0_2)  c_e_BusBlock_balance(BB_h2_0_2)_  -1
    flow(BB_electricity_BB_electricity_liion_battery_0_0)  c_e_BusBlock_balance(BB_electricity_0_0)_  -1
    flow(BB_hydro_ror_BB_electricity_0_0)  c_e_BusBlock_balance(BB_electricity_0_0)_  1
    flow(BB_h2_bpchp_BB_electricity_0_0)  c_e_BusBlock_balance(BB_electricity_0_0)_  1
    flow(BB_solar_pv_BB_electricity_0_0)  c_e_BusBlock_balance(BB_electricity_0_0)_  1
    flow(BB_ch4_bpchp_BB_electricity_0_0)  c_e_BusBlock_balance(BB_electricity_0_0)_  1
    flow(BB_B_electricity_transmission_BB_electricity_0_0)  c_e_BusBlock_balance(BB_electricity_0_0)_  1
    flow(BB_electricity_BB_electricity_curtailment_0_0)  c_e_BusBlock_balance(BB_electricity_0_0)_  -1
    flow(BB_electricity_BB_electricity_electrolyzer_0_0)  c_e_BusBlock_balance(BB_electricity_0_0)_  -1
    flow(BB_electricity_BB_electricity_export_0_0)  c_e_BusBlock_balance(BB_electricity_0_0)_  -1
    flow(BB_electricity_BB_electricity_heatpump_large_0_0)  c_e_BusBlock_balance(BB_electricity_0_0)_  -1
    flow(BB_electricity_BB_electricity_heatpump_small_0_0)  c_e_BusBlock_balance(BB_electricity_0_0)_  -1
    flow(BB_electricity_BB_electricity_pth_0_0)  c_e_BusBlock_balance(BB_electricity_0_0)_  -1
    flow(BB_electricity_BB_B_electricity_transmission_0_0)  c_e_BusBlock_balance(BB_electricity_0_0)_  -1
    flow(BB_electricity_BB_electricity_liion_battery_0_1)  c_e_BusBlock_balance(BB_electricity_0_1)_  -1
    flow(BB_hydro_ror_BB_electricity_0_1)  c_e_BusBlock_balance(BB_electricity_0_1)_  1
    flow(BB_h2_bpchp_BB_electricity_0_1)  c_e_BusBlock_balance(BB_electricity_0_1)_  1
    flow(BB_solar_pv_BB_electricity_0_1)  c_e_BusBlock_balance(BB_electricity_0_1)_  1
    flow(BB_ch4_bpchp_BB_electricity_0_1)  c_e_BusBlock_balance(BB_electricity_0_1)_  1
    flow(BB_B_electricity_transmission_BB_electricity_0_1)  c_e_BusBlock_balance(BB_electricity_0_1)_  1
    flow(BB_electricity_BB_electricity_curtailment_0_1)  c_e_BusBlock_balance(BB_electricity_0_1)_  -1
    flow(BB_electricity_BB_electricity_electrolyzer_0_1)  c_e_BusBlock_balance(BB_electricity_0_1)_  -1
    flow(BB_electricity_BB_electricity_export_0_1)  c_e_BusBlock_balance(BB_electricity_0_1)_  -1
    flow(BB_electricity_BB_electricity_heatpump_large_0_1)  c_e_BusBlock_balance(BB_electricity_0_1)_  -1
    flow(BB_electricity_BB_electricity_heatpump_small_0_1)  c_e_BusBlock_balance(BB_electricity_0_1)_  -1
    flow(BB_electricity_BB_electricity_pth_0_1)  c_e_BusBlock_balance(BB_electricity_0_1)_  -1
    flow(BB_electricity_BB_B_electricity_transmission_0_1)  c_e_BusBlock_balance(BB_electricity_0_1)_  -1
    flow(BB_electricity_BB_electricity_liion_battery_0_2)  c_e_BusBlock_balance(BB_electricity_0_2)_  -1
    flow(BB_hydro_ror_BB_electricity_0_2)  c_e_BusBlock_balance(BB_electricity_0_2)_  1
    flow(BB_h2_bpchp_BB_electricity_0_2)  c_e_BusBlock_balance(BB_electricity_0_2)_  1
    flow(BB_solar_pv_BB_electricity_0_2)  c_e_BusBlock_balance(BB_electricity_0_2)_  1
    flow(BB_ch4_bpchp_BB_electricity_0_2)  c_e_BusBlock_balance(BB_electricity_0_2)_  1
    flow(BB_B_electricity_transmission_BB_electricity_0_2)  c_e_BusBlock_balance(BB_electricity_0_2)_  1
    flow(BB_electricity_BB_electricity_curtailment_0_2)  c_e_BusBlock_balance(BB_electricity_0_2)_  -1
    flow(BB_electricity_BB_electricity_electrolyzer_0_2)  c_e_BusBlock_balance(BB_electricity_0_2)_  -1
    flow(BB_electricity_BB_electricity_export_0_2)  c_e_BusBlock_balance(BB_electricity_0_2)_  -1
    flow(BB_electricity_BB_electricity_heatpump_large_0_2)  c_e_BusBlock_balance(BB_electricity_0_2)_  -1
    flow(BB_electricity_BB_electricity_heatpump_small_0_2)  c_e_BusBlock_balance(BB_electricity_0_2)_  -1
    flow(BB_electricity_BB_electricity_pth_0_2)  c_e_BusBlock_balance(BB_electricity_0_2)_  -1
    flow(BB_electricity_BB_B_electricity_transmission_0_2)  c_e_BusBlock_balance(BB_electricity_0_2)_  -1
    flow(B_electricity_B_electricity_liion_battery_0_0)  c_e_BusBlock_balance(B_electricity_0_0)_  -1
    flow(B_hydro_ror_B_electricity_0_0)  c_e_BusBlock_balance(B_electricity_0_0)_  1
    flow(B_h2_bpchp_B_electricity_0_0)  c_e_BusBlock_balance(B_electricity_0_0)_  1
    flow(B_ch4_bpchp_B_electricity_0_0)  c_e_BusBlock_balance(B_electricity_0_0)_  1
    flow(B_solar_pv_B_electricity_0_0)  c_e_BusBlock_balance(B_electricity_0_0)_  1
    flow(BB_B_electricity_transmission_B_electricity_0_0)  c_e_BusBlock_balance(B_electricity_0_0)_  1
    flow(B_electricity_B_electricity_curtailment_0_0)  c_e_BusBlock_balance(B_electricity_0_0)_  -1
    flow(B_electricity_B_electricity_electrolyzer_0_0)  c_e_BusBlock_balance(B_electricity_0_0)_  -1
    flow(B_electricity_B_electricity_export_0_0)  c_e_BusBlock_balance(B_electricity_0_0)_  -1
    flow(B_electricity_B_electricity_heatpump_large_0_0)  c_e_BusBlock_balance(B_electricity_0_0)_  -1
    flow(B_electricity_B_electricity_heatpump_small_0_0)  c_e_BusBlock_balance(B_electricity_0_0)_  -1
    flow(B_electricity_B_electricity_pth_0_0)  c_e_BusBlock_balance(B_electricity_0_0)_  -1
    flow(B_electricity_BB_B_electricity_transmission_0_0)  c_e_BusBlock_balance(B_electricity_0_0)_  -1
    flow(B_electricity_B_electricity_liion_battery_0_1)  c_e_BusBlock_balance(B_electricity_0_1)_  -1
    flow(B_hydro_ror_B_electricity_0_1)  c_e_BusBlock_balance(B_electricity_0_1)_  1
    flow(B_h2_bpchp_B_electricity_0_1)  c_e_BusBlock_balance(B_electricity_0_1)_  1
    flow(B_ch4_bpchp_B_electricity_0_1)  c_e_BusBlock_balance(B_electricity_0_1)_  1
    flow(B_solar_pv_B_electricity_0_1)  c_e_BusBlock_balance(B_electricity_0_1)_  1
    flow(BB_B_electricity_transmission_B_electricity_0_1)  c_e_BusBlock_balance(B_electricity_0_1)_  1
    flow(B_electricity_B_electricity_curtailment_0_1)  c_e_BusBlock_balance(B_electricity_0_1)_  -1
    flow(B_electricity_B_electricity_electrolyzer_0_1)  c_e_BusBlock_balance(B_electricity_0_1)_  -1
    flow(B_electricity_B_electricity_export_0_1)  c_e_BusBlock_balance(B_electricity_0_1)_  -1
    flow(B_electricity_B_electricity_heatpump_large_0_1)  c_e_BusBlock_balance(B_electricity_0_1)_  -1
    flow(B_electricity_B_electricity_heatpump_small_0_1)  c_e_BusBlock_balance(B_electricity_0_1)_  -1
    flow(B_electricity_B_electricity_pth_0_1)  c_e_BusBlock_balance(B_electricity_0_1)_  -1
    flow(B_electricity_BB_B_electricity_transmission_0_1)  c_e_BusBlock_balance(B_electricity_0_1)_  -1
    flow(B_electricity_B_electricity_liion_battery_0_2)  c_e_BusBlock_balance(B_electricity_0_2)_  -1
    flow(B_hydro_ror_B_electricity_0_2)  c_e_BusBlock_balance(B_electricity_0_2)_  1
    flow(B_h2_bpchp_B_electricity_0_2)  c_e_BusBlock_balance(B_electricity_0_2)_  1
    flow(B_ch4_bpchp_B_electricity_0_2)  c_e_BusBlock_balance(B_electricity_0_2)_  1
    flow(B_solar_pv_B_electricity_0_2)  c_e_BusBlock_balance(B_electricity_0_2)_  1
    flow(BB_B_electricity_transmission_B_electricity_0_2)  c_e_BusBlock_balance(B_electricity_0_2)_  1
    flow(B_electricity_B_electricity_curtailment_0_2)  c_e_BusBlock_balance(B_electricity_0_2)_  -1
    flow(B_electricity_B_electricity_electrolyzer_0_2)  c_e_BusBlock_balance(B_electricity_0_2)_  -1
    flow(B_electricity_B_electricity_export_0_2)  c_e_BusBlock_balance(B_electricity_0_2)_  -1
    flow(B_electricity_B_electricity_heatpump_large_0_2)  c_e_BusBlock_balance(B_electricity_0_2)_  -1
    flow(B_electricity_B_electricity_heatpump_small_0_2)  c_e_BusBlock_balance(B_electricity_0_2)_  -1
    flow(B_electricity_B_electricity_pth_0_2)  c_e_BusBlock_balance(B_electricity_0_2)_  -1
    flow(B_electricity_BB_B_electricity_transmission_0_2)  c_e_BusBlock_balance(B_electricity_0_2)_  -1
    GenericInvestmentStorageBlock_storage_content(BB_heat_decentral_storage_0)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(B_h2_cavern_0)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(B_heat_decentral_storage_0)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(BB_h2_cavern_0)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(B_electricity_liion_battery_0)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(BB_heat_central_storage_0)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(BB_electricity_liion_battery_0)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(B_heat_central_storage_0)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(BB_heat_decentral_storage_1)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(BB_heat_decentral_storage_2)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(B_h2_cavern_1)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(B_h2_cavern_2)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(B_heat_decentral_storage_1)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(B_heat_decentral_storage_2)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(BB_h2_cavern_1)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(BB_h2_cavern_2)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(B_electricity_liion_battery_1)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(B_electricity_liion_battery_2)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(BB_heat_central_storage_1)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(BB_heat_central_storage_2)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(BB_electricity_liion_battery_1)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(BB_electricity_liion_battery_2)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(B_heat_central_storage_1)  OBJ       0
    GenericInvestmentStorageBlock_storage_content(B_heat_central_storage_2)  OBJ       0
RHS
    RHS1      c_e_BusBlock_balance(B_ch4_0_0)_  7.8515981735159812e+00
    RHS1      c_e_BusBlock_balance(B_ch4_0_1)_  7.8515981735159812e+00
    RHS1      c_e_BusBlock_balance(B_ch4_0_2)_  7.8515981735159812e+00
    RHS1      c_e_BusBlock_balance(BB_heat_decentral_0_0)_  1.8596079764465726e+03
    RHS1      c_e_BusBlock_balance(BB_heat_decentral_0_1)_  1.9225392510894374e+03
    RHS1      c_e_BusBlock_balance(BB_heat_decentral_0_2)_  1.9166984827317995e+03
    RHS1      c_e_BusBlock_balance(B_heat_central_0_0)_  9.7152893176598377e+02
    RHS1      c_e_BusBlock_balance(B_heat_central_0_1)_  1.0087854912909980e+03
    RHS1      c_e_BusBlock_balance(B_heat_central_0_2)_  1.0198332743176152e+03
    RHS1      c_e_BusBlock_balance(BB_ch4_0_0)_  4.0516655251141549e+02
    RHS1      c_e_BusBlock_balance(BB_ch4_0_1)_  4.0516655251141549e+02
    RHS1      c_e_BusBlock_balance(BB_ch4_0_2)_  4.0516655251141549e+02
    RHS1      c_e_BusBlock_balance(B_h2_0_0)_  4.0842899543378996e+02
    RHS1      c_e_BusBlock_balance(B_h2_0_1)_  4.0842899543378996e+02
    RHS1      c_e_BusBlock_balance(B_h2_0_2)_  4.0842899543378996e+02
    RHS1      c_e_BusBlock_balance(B_heat_decentral_0_0)_  9.6374615794186639e+02
    RHS1      c_e_BusBlock_balance(B_heat_decentral_0_1)_  9.9128637464448957e+02
    RHS1      c_e_BusBlock_balance(B_heat_decentral_0_2)_  9.8823981819622543e+02
    RHS1      c_e_BusBlock_balance(BB_heat_central_0_0)_  4.8525325405343892e+02
    RHS1      c_e_BusBlock_balance(BB_heat_central_0_1)_  5.0691526886667145e+02
    RHS1      c_e_BusBlock_balance(BB_heat_central_0_2)_  5.1187411717296317e+02
    RHS1      c_e_BusBlock_balance(BB_h2_0_0)_  6.9269178082191775e+02
    RHS1      c_e_BusBlock_balance(BB_h2_0_1)_  6.9269178082191775e+02
    RHS1      c_e_BusBlock_balance(BB_h2_0_2)_  6.9269178082191775e+02
    RHS1      c_e_BusBlock_balance(BB_electricity_0_0)_  3.8061154334889179e+03
    RHS1      c_e_BusBlock_balance(BB_electricity_0_1)_  3.2962499375723355e+03
    RHS1      c_e_BusBlock_balance(BB_electricity_0_2)_  3.1106955399557323e+03
    RHS1      c_e_BusBlock_balance(B_electricity_0_0)_  9.3135314475892687e+02
    RHS1      c_e_BusBlock_balance(B_electricity_0_1)_  9.1367662701403947e+02
    RHS1      c_e_BusBlock_balance(B_electricity_0_2)_  8.7462346482229975e+02
BOUNDS
 UP BND1      flow(B_ch4_import_B_ch4_0_0)  1000000000
 UP BND1      flow(B_ch4_import_B_ch4_0_1)  1000000000
 UP BND1      flow(B_ch4_import_B_ch4_0_2)  1000000000
 UP BND1      flow(BB_ch4_import_BB_ch4_0_0)  1000000000
 UP BND1      flow(BB_ch4_import_BB_ch4_0_1)  1000000000
 UP BND1      flow(BB_ch4_import_BB_ch4_0_2)  1000000000
 UP BND1      flow(B_ch4_shortage_B_ch4_0_0)  1000000000
 UP BND1      flow(B_ch4_shortage_B_ch4_0_1)  1000000000
 UP BND1      flow(B_ch4_shortage_B_ch4_0_2)  1000000000
 UP BND1      flow(BB_ch4_shortage_BB_ch4_0_0)  1000000000
 UP BND1      flow(BB_ch4_shortage_BB_ch4_0_1)  1000000000
 UP BND1      flow(BB_ch4_shortage_BB_ch4_0_2)  1000000000
 FX BND1      flow(B_electricity_import_B_electricity_0_0)  0
 FX BND1      flow(B_electricity_import_B_electricity_0_1)  0
 FX BND1      flow(B_electricity_import_B_electricity_0_2)  0
 UP BND1      flow(BB_electricity_import_BB_electricity_0_0)  15235
 UP BND1      flow(BB_electricity_import_BB_electricity_0_1)  15235
 UP BND1      flow(BB_electricity_import_BB_electricity_0_2)  15235
 UP BND1      flow(B_electricity_shortage_B_electricity_0_0)  1000000000
 UP BND1      flow(B_electricity_shortage_B_electricity_0_1)  1000000000
 UP BND1      flow(B_electricity_shortage_B_electricity_0_2)  1000000000
 UP BND1      flow(BB_electricity_shortage_BB_electricity_0_0)  1000000000
 UP BND1      flow(BB_electricity_shortage_BB_electricity_0_1)  1000000000
 UP BND1      flow(BB_electricity_shortage_BB_electricity_0_2)  1000000000
 UP BND1      flow(B_h2_import_B_h2_0_0)  1000000000
 UP BND1      flow(B_h2_import_B_h2_0_1)  1000000000
 UP BND1      flow(B_h2_import_B_h2_0_2)  1000000000
 UP BND1      flow(BB_h2_import_BB_h2_0_0)  1000000000
 UP BND1      flow(BB_h2_import_BB_h2_0_1)  1000000000
 UP BND1      flow(BB_h2_import_BB_h2_0_2)  1000000000
 UP BND1      flow(B_h2_shortage_B_h2_0_0)  1000000000
 UP BND1      flow(B_h2_shortage_B_h2_0_1)  1000000000
 UP BND1      flow(B_h2_shortage_B_h2_0_2)  1000000000
 UP BND1      flow(BB_h2_shortage_BB_h2_0_0)  1000000000
 UP BND1      flow(BB_h2_shortage_BB_h2_0_1)  1000000000
 UP BND1      flow(BB_h2_shortage_BB_h2_0_2)  1000000000
 UP BND1      flow(B_heat_central_shortage_B_heat_central_0_0)  1000000000
 UP BND1      flow(B_heat_central_shortage_B_heat_central_0_1)  1000000000
 UP BND1      flow(B_heat_central_shortage_B_heat_central_0_2)  1000000000
 UP BND1      flow(BB_heat_central_shortage_BB_heat_central_0_0)  1000000000
 UP BND1      flow(BB_heat_central_shortage_BB_heat_central_0_1)  1000000000
 UP BND1      flow(BB_heat_central_shortage_BB_heat_central_0_2)  1000000000
 UP BND1      flow(B_heat_decentral_shortage_B_heat_decentral_0_0)  1000000000
 UP BND1      flow(B_heat_decentral_shortage_B_heat_decentral_0_1)  1000000000
 UP BND1      flow(B_heat_decentral_shortage_B_heat_decentral_0_2)  1000000000
 UP BND1      flow(BB_heat_decentral_shortage_BB_heat_decentral_0_0)  1000000000
 UP BND1      flow(BB_heat_decentral_shortage_BB_heat_decentral_0_1)  1000000000
 UP BND1      flow(BB_heat_decentral_shortage_BB_heat_decentral_0_2)  1000000000
 UP BND1      flow(B_ch4_B_ch4_excess_0_0)  1000000000
 UP BND1      flow(B_ch4_B_ch4_export_0_0)  1000000000
 UP BND1      flow(B_ch4_B_ch4_excess_0_1)  1000000000
 UP BND1      flow(B_ch4_B_ch4_export_0_1)  1000000000
 UP BND1      flow(B_ch4_B_ch4_excess_0_2)  1000000000
 UP BND1      flow(B_ch4_B_ch4_export_0_2)  1000000000
 UP BND1      flow(BB_heat_decentral_BB_heat_decentral_excess_0_0)  1000000000
 UP BND1      flow(BB_heat_decentral_BB_heat_decentral_excess_0_1)  1000000000
 UP BND1      flow(BB_heat_decentral_BB_heat_decentral_excess_0_2)  1000000000
 UP BND1      flow(B_heat_central_B_heat_central_excess_0_0)  1000000000
 UP BND1      flow(B_heat_central_B_heat_central_excess_0_1)  1000000000
 UP BND1      flow(B_heat_central_B_heat_central_excess_0_2)  1000000000
 UP BND1      flow(BB_ch4_BB_ch4_excess_0_0)  1000000000
 UP BND1      flow(BB_ch4_BB_ch4_export_0_0)  1000000000
 UP BND1      flow(BB_ch4_BB_ch4_excess_0_1)  1000000000
 UP BND1      flow(BB_ch4_BB_ch4_export_0_1)  1000000000
 UP BND1      flow(BB_ch4_BB_ch4_excess_0_2)  1000000000
 UP BND1      flow(BB_ch4_BB_ch4_export_0_2)  1000000000
 UP BND1      flow(B_h2_B_h2_excess_0_0)  1000000000
 UP BND1      flow(B_h2_B_h2_export_0_0)  1000000000
 UP BND1      flow(B_h2_B_h2_excess_0_1)  1000000000
 UP BND1      flow(B_h2_B_h2_export_0_1)  1000000000
 UP BND1      flow(B_h2_B_h2_excess_0_2)  1000000000
 UP BND1      flow(B_h2_B_h2_export_0_2)  1000000000
 UP BND1      flow(B_heat_decentral_B_heat_decentral_excess_0_0)  1000000000
 UP BND1      flow(B_heat_decentral_B_heat_decentral_excess_0_1)  1000000000
 UP BND1      flow(B_heat_decentral_B_heat_decentral_excess_0_2)  1000000000
 UP BND1      flow(BB_heat_central_BB_heat_central_excess_0_0)  1000000000
 UP BND1      flow(BB_heat_central_BB_heat_central_excess_0_1)  1000000000
 UP BND1      flow(BB_heat_central_BB_heat_central_excess_0_2)  1000000000
 UP BND1      flow(BB_h2_BB_h2_excess_0_0)  1000000000
 UP BND1      flow(BB_h2_BB_h2_export_0_0)  1000000000
 UP BND1      flow(BB_h2_BB_h2_excess_0_1)  1000000000
 UP BND1      flow(BB_h2_BB_h2_export_0_1)  1000000000
 UP BND1      flow(BB_h2_BB_h2_excess_0_2)  1000000000
 UP BND1      flow(BB_h2_BB_h2_export_0_2)  1000000000
 UP BND1      flow(BB_B_electricity_transmission_BB_electricity_0_0)  3000
 UP BND1      flow(BB_electricity_BB_electricity_curtailment_0_0)  1000000000
 UP BND1      flow(BB_electricity_BB_electricity_export_0_0)  15235
 UP BND1      flow(BB_B_electricity_transmission_BB_electricity_0_1)  3000
 UP BND1      flow(BB_electricity_BB_electricity_curtailment_0_1)  1000000000
 UP BND1      flow(BB_electricity_BB_electricity_export_0_1)  15235
 UP BND1      flow(BB_B_electricity_transmission_BB_electricity_0_2)  3000
 UP BND1      flow(BB_electricity_BB_electricity_curtailment_0_2)  1000000000
 UP BND1      flow(BB_electricity_BB_electricity_export_0_2)  15235
 UP BND1      flow(BB_B_electricity_transmission_B_electricity_0_0)  3000
 UP BND1      flow(B_electricity_B_electricity_curtailment_0_0)  1000000000
 FX BND1      flow(B_electricity_B_electricity_export_0_0)  0
 UP BND1      flow(BB_B_electricity_transmission_B_electricity_0_1)  3000
 UP BND1      flow(B_electricity_B_electricity_curtailment_0_1)  1000000000
 FX BND1      flow(B_electricity_B_electricity_export_0_1)  0
 UP BND1      flow(BB_B_electricity_transmission_B_electricity_0_2)  3000
 UP BND1      flow(B_electricity_B_electricity_curtailment_0_2)  1000000000
 FX BND1      flow(B_electricity_B_electricity_export_0_2)  0
ENDATA
