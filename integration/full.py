#  Файл создан tNavigator v21.3-2272-g351282a5db0.
#  Copyright (C) RFDynamics 2005-2021.
#  Все права защищены.

# This file is MACHINE GENERATED! Do not edit.

#api_version=v0.0.36

from __main__.tnav.workflow import *
from tnav_debug_utilities import *
from datetime import datetime, timedelta


declare_workflow (workflow_name="full",
      variables=[{"name" : "H", "type" : "real", "min" : 20, "max" : 50, "values" : [], "distribution_type" : "Uniform", "discrete_distr_values" : [], "discrete_distr_probabilities" : [], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "PORO", "type" : "real", "min" : 0.11, "max" : 0.15, "values" : [], "distribution_type" : "Uniform", "discrete_distr_values" : [], "discrete_distr_probabilities" : [], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "GLOBAL_GNK", "type" : "real", "min" : 2230, "max" : 2250, "values" : [], "distribution_type" : "Uniform", "discrete_distr_values" : [], "discrete_distr_probabilities" : [], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "GLOBAL_VNK", "type" : "real", "min" : 2260, "max" : 2280, "values" : [], "distribution_type" : "Uniform", "discrete_distr_values" : [], "discrete_distr_probabilities" : [], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "TEMPERATURE", "type" : "integer", "min" : 50, "max" : 70, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [], "discrete_distr_probabilities" : [], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "PRESSURE", "type" : "integer", "min" : 210, "max" : 240, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [], "discrete_distr_probabilities" : [], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "DENSITY_OIL", "type" : "integer", "min" : 830, "max" : 870, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [], "discrete_distr_probabilities" : [], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "DENSITY_WATER", "type" : "integer", "min" : 1010, "max" : 1050, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [], "discrete_distr_probabilities" : [], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "DENSITY_GAS", "type" : "real", "min" : 1.1, "max" : 1.3, "values" : [], "distribution_type" : "Uniform", "discrete_distr_values" : [], "discrete_distr_probabilities" : [], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "SELECTION_OIL", "type" : "real", "min" : 20, "max" : 90, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [20, 30, 40, 50, 60, 70, 80, 90], "discrete_distr_probabilities" : [12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "SELECTION_GAS", "type" : "real", "min" : 20, "max" : 90, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [20, 30, 40, 50, 60, 70, 80, 90], "discrete_distr_probabilities" : [12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "WP_MAX_OIL", "type" : "real", "min" : 5, "max" : 12, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [5, 6, 7, 8, 9, 10, 11, 12], "discrete_distr_probabilities" : [12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "WP_MAX_GAS", "type" : "real", "min" : 5, "max" : 12, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [5, 6, 7, 8, 9, 10, 11, 12], "discrete_distr_probabilities" : [12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "WT_MAX_OIL", "type" : "real", "min" : 1, "max" : 4, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [1, 2, 3, 4], "discrete_distr_probabilities" : [25, 25, 25, 25], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "WT_MAX_GAS", "type" : "real", "min" : 1, "max" : 4, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [1, 2, 3, 4], "discrete_distr_probabilities" : [25, 25, 25, 25], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "MOBIL_BU_OIL", "type" : "real", "min" : 1, "max" : 3, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [1, 2, 3], "discrete_distr_probabilities" : [33.34, 33.33, 33.33], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "MOBIL_BU_GAS", "type" : "real", "min" : 1, "max" : 3, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [1, 2, 3], "discrete_distr_probabilities" : [33.34, 33.33, 33.33], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "DEP_GAS", "type" : "real", "min" : 2, "max" : 45, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [2, 5, 10, 15, 20, 25, 30, 35, 40, 45], "discrete_distr_probabilities" : [10, 10, 10, 10, 10, 10, 10, 10, 10, 10], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}, {"name" : "DEP_OIL", "type" : "real", "min" : 2, "max" : 45, "values" : [], "distribution_type" : "Discrete", "discrete_distr_values" : [2, 5, 10, 15, 20, 25, 30, 35, 40, 45], "discrete_distr_probabilities" : [10, 10, 10, 10, 10, 10, 10, 10, 10, 10], "initial_distribution" : [], "truncated_mean" : 0, "truncated_sigma" : 0}])


full_variables = {
"H" : 20,
"PORO" : 0.13,
"GLOBAL_GNK" : 2240,
"GLOBAL_VNK" : 2270,
"TEMPERATURE" : 60,
"PRESSURE" : 225,
"DENSITY_OIL" : 850,
"DENSITY_WATER" : 1030,
"DENSITY_GAS" : 1.2,
"SELECTION_OIL" : 70,
"SELECTION_GAS" : 70,
"WP_MAX_OIL" : 5,
"WP_MAX_GAS" : 5,
"WT_MAX_OIL" : 4,
"WT_MAX_GAS" : 2,
"MOBIL_BU_OIL" : 1,
"MOBIL_BU_GAS" : 1,
"DEP_GAS" : 2,
"DEP_OIL" : 2
}

def full (variables = full_variables):
    pass
    check_launch_method ()

    H = variables["H"]
    PORO = variables["PORO"]
    GLOBAL_GNK = variables["GLOBAL_GNK"]
    GLOBAL_VNK = variables["GLOBAL_VNK"]
    TEMPERATURE = variables["TEMPERATURE"]
    PRESSURE = variables["PRESSURE"]
    DENSITY_OIL = variables["DENSITY_OIL"]
    DENSITY_WATER = variables["DENSITY_WATER"]
    DENSITY_GAS = variables["DENSITY_GAS"]
    SELECTION_OIL = variables["SELECTION_OIL"]
    SELECTION_GAS = variables["SELECTION_GAS"]
    WP_MAX_OIL = variables["WP_MAX_OIL"]
    WP_MAX_GAS = variables["WP_MAX_GAS"]
    WT_MAX_OIL = variables["WT_MAX_OIL"]
    WT_MAX_GAS = variables["WT_MAX_GAS"]
    MOBIL_BU_OIL = variables["MOBIL_BU_OIL"]
    MOBIL_BU_GAS = variables["MOBIL_BU_GAS"]
    DEP_GAS = variables["DEP_GAS"]
    DEP_OIL = variables["DEP_OIL"]

    begin_user_imports ()
    import numpy as np
    import json
    import shutil
    import os
    import runpy
    end_user_imports ()

    if False:
        begin_wf_item (index = 1, is_custom_code = True, name = "Случайное глобальное число")
        np.random.seed(11)
        end_wf_item (index = 1)


    if False:
        begin_wf_item (index = 2, is_custom_code = True, name = "Библиотеки тут")
        exec ("""
#import numpy as np
#from scipy.interpolate import interp1d
#from scipy.ndimage import gaussian_filter
#from scipy.ndimage import median_filter
""")
        end_wf_item (index = 2)


    begin_wf_item (index = 3, name = "Разломы")
    workflow_folder ()
    if True:
        pass



        begin_wf_item (index = 4)
        polygon_import_txt_format (use_tags_to_assign=False,
              tags_to_assign=[],
              use_folder=False,
              folder="",
              splitter=True,
              files_table=[{"file_name" : "frac_line/f1_bot_1.txt", "prefix" : "f1_bot_1"}, {"file_name" : "frac_line/f1_top_1.txt", "prefix" : "f1_top_1"}, {"file_name" : "frac_line/f2_bot_1.txt", "prefix" : "f2_bot_1"}, {"file_name" : "frac_line/f2_top_1.txt", "prefix" : "f2_top_1"}, {"file_name" : "frac_line/f3_bot_1.txt", "prefix" : "f3_bot_1"}, {"file_name" : "frac_line/f3_top_1.txt", "prefix" : "f3_top_1"}, {"file_name" : "frac_line/f4_bot_1.txt", "prefix" : "f4_bot_1"}, {"file_name" : "frac_line/f4_top_1.txt", "prefix" : "f4_top_1"}, {"file_name" : "frac_line/f5_bot_1.txt", "prefix" : "f5_bot_1"}, {"file_name" : "frac_line/f5_top_1.txt", "prefix" : "f5_top_1"}, {"file_name" : "frac_line/f6_bot_1.txt", "prefix" : "f6_bot_1"}, {"file_name" : "frac_line/f6_top_1.txt", "prefix" : "f6_top_1"}],
              splitter2=True,
              splitter3=True,
              file_datum_info=CrsInfo (crs_type="not_specified",
              crs_code=None,
              crs_name="",
              crs_proj_string=None,
              datum_name=None,
              datum_bounds_inited=False,
              datum_bounds_min_x=0,
              datum_bounds_max_x=0,
              datum_bounds_min_y=0,
              datum_bounds_max_y=0,
              datum_is_in_proj4=False),
              is_closed=True,
              invert_z=False,
              units_system="si",
              use_xy_units=True,
              xy_units="metres",
              use_z_units=True,
              z_units="metres")
        end_wf_item (index = 4)


        begin_wf_item (index = 5)
        workflow_folder ()
        if True:
            pass



            if False:
                begin_wf_item (index = 6)
                polygon_set_z_by_depth (polygons=[{"polygon" : find_object (name="f1_top",
                      type="Curve3d"), "depth" : 1800}, {"polygon" : find_object (name="f1_bot",
                      type="Curve3d"), "depth" : 2800}, {"polygon" : find_object (name="f2_top",
                      type="Curve3d"), "depth" : 1800}, {"polygon" : find_object (name="f2_bot",
                      type="Curve3d"), "depth" : 2800}, {"polygon" : find_object (name="f3_top",
                      type="Curve3d"), "depth" : 1800}, {"polygon" : find_object (name="f3_bot",
                      type="Curve3d"), "depth" : 2800}, {"polygon" : find_object (name="f4_top",
                      type="Curve3d"), "depth" : 1800}, {"polygon" : find_object (name="f4_bot",
                      type="Curve3d"), "depth" : 2800}, {"polygon" : find_object (name="f5_top",
                      type="Curve3d"), "depth" : 1800}, {"polygon" : find_object (name="f5_bot",
                      type="Curve3d"), "depth" : 2800}, {"polygon" : find_object (name="f6_top",
                      type="Curve3d"), "depth" : 1800}, {"polygon" : find_object (name="f6_bot",
                      type="Curve3d"), "depth" : 2800}])
                end_wf_item (index = 6)


            if False:
                begin_wf_item (index = 7)
                fault_create_by_polygons (fault=find_object (name="Fault1",
                      type="Fault3d"),
                      polygons_table=[{"polygon" : find_object (name="f1_top",
                      type="Curve3d")}, {"polygon" : find_object (name="f1_bot",
                      type="Curve3d")}],
                      reorder_sticks=False)
                end_wf_item (index = 7)


            if False:
                begin_wf_item (index = 8)
                fault_create_by_polygons (fault=find_object (name="Fault2",
                      type="Fault3d"),
                      polygons_table=[{"polygon" : find_object (name="f2_top",
                      type="Curve3d")}, {"polygon" : find_object (name="f2_bot",
                      type="Curve3d")}],
                      reorder_sticks=False)
                end_wf_item (index = 8)


            if False:
                begin_wf_item (index = 9)
                fault_create_by_polygons (fault=find_object (name="Fault3",
                      type="Fault3d"),
                      polygons_table=[{"polygon" : find_object (name="f3_top",
                      type="Curve3d")}, {"polygon" : find_object (name="f3_bot",
                      type="Curve3d")}],
                      reorder_sticks=False)
                end_wf_item (index = 9)


            if False:
                begin_wf_item (index = 10)
                fault_create_by_polygons (fault=find_object (name="Fault4",
                      type="Fault3d"),
                      polygons_table=[{"polygon" : find_object (name="f4_top",
                      type="Curve3d")}, {"polygon" : find_object (name="f4_bot",
                      type="Curve3d")}],
                      reorder_sticks=False)
                end_wf_item (index = 10)


            if False:
                begin_wf_item (index = 11)
                fault_create_by_polygons (fault=find_object (name="Fault5",
                      type="Fault3d"),
                      polygons_table=[{"polygon" : find_object (name="f5_top",
                      type="Curve3d")}, {"polygon" : find_object (name="f5_bot",
                      type="Curve3d")}],
                      reorder_sticks=False)
                end_wf_item (index = 11)


            if False:
                begin_wf_item (index = 12)
                fault_create_by_polygons (fault=find_object (name="Fault6",
                      type="Fault3d"),
                      polygons_table=[{"polygon" : find_object (name="f6_top",
                      type="Curve3d")}, {"polygon" : find_object (name="f6_bot",
                      type="Curve3d")}],
                      reorder_sticks=False)
                end_wf_item (index = 12)


            if False:
                begin_wf_item (index = 13)
                fault_smooth (source_surface=find_object (name="Fault1",
                      type="Fault3d"),
                      smoothed_surface=find_object (name="Fault1",
                      type="Fault3d"),
                      smoothing_method="splines",
                      subdivision=8,
                      moving_average_radius=0.05)
                end_wf_item (index = 13)


            if False:
                begin_wf_item (index = 14)
                fault_smooth (source_surface=find_object (name="Fault2",
                      type="Fault3d"),
                      smoothed_surface=find_object (name="Fault2",
                      type="Fault3d"),
                      smoothing_method="splines",
                      subdivision=8,
                      moving_average_radius=0.05)
                end_wf_item (index = 14)


            if False:
                begin_wf_item (index = 15)
                fault_smooth (source_surface=find_object (name="Fault3",
                      type="Fault3d"),
                      smoothed_surface=find_object (name="Fault3",
                      type="Fault3d"),
                      smoothing_method="splines",
                      subdivision=8,
                      moving_average_radius=0.05)
                end_wf_item (index = 15)


            if False:
                begin_wf_item (index = 16)
                fault_smooth (source_surface=find_object (name="Fault4",
                      type="Fault3d"),
                      smoothed_surface=find_object (name="Fault4",
                      type="Fault3d"),
                      smoothing_method="splines",
                      subdivision=8,
                      moving_average_radius=0.05)
                end_wf_item (index = 16)


            if False:
                begin_wf_item (index = 17)
                fault_smooth (source_surface=find_object (name="Fault5",
                      type="Fault3d"),
                      smoothed_surface=find_object (name="Fault5",
                      type="Fault3d"),
                      smoothing_method="splines",
                      subdivision=8,
                      moving_average_radius=0.05)
                end_wf_item (index = 17)


            if False:
                begin_wf_item (index = 18)
                fault_smooth (source_surface=find_object (name="Fault6",
                      type="Fault3d"),
                      smoothed_surface=find_object (name="Fault6",
                      type="Fault3d"),
                      smoothing_method="splines",
                      subdivision=8,
                      moving_average_radius=0.05)
                end_wf_item (index = 18)



        end_wf_item (index = 5)



    end_wf_item (index = 3)


    if False:
        begin_wf_item (index = 21, name = "Сетка")
        workflow_folder ()
        if True:
            pass



            if False:
                begin_wf_item (index = 22)
                horizon_calculator (result_horizon=find_object (name="bot",
                      type="Horizon"),
                      use_polygon=False,
                      polygon=find_object (name="float_line_1_top",
                      type="Curve3d"),
                      use_initial_geometry=False,
                      grid_2d_gui_data=Grid2DAutodetector (disabled=True,
                      autodetect_angle=True,
                      autodetect_sample_object=find_object (name="top",
                      type="Horizon"),
                      margin=0),
                      grid_2d=Grid2D (step_x=100,
                      step_y=100,
                      area=Rectangle (origin_x=0,
                      origin_y=0,
                      size_x=36100,
                      size_y=56400,
                      angle=0)),
                      formula=resolve_variables_in_string (string_with_variables="top + @H@",
                      variables=variables),
                      variables=variables)
                end_wf_item (index = 22)


            if False:
                begin_wf_item (index = 23)
                grid_3d_create_by_horizons_with_faults (grid=find_object (name="test_grid",
                      type="Grid3d"),
                      use_one_layering_column=True,
                      use_individual_minimum_zone_thickness_column=False,
                      proportions_table=find_object (name="",
                      type="Table"),
                      horizons_table=[{"horizon" : find_object (name="top",
                      type="Horizon"), "zone" : "Zone1", "partition_type" : "proportional", "counts_step" : 50, "individual_minimum_zone_thickness" : 0, "source_marker" : None, "layering_horizon" : None, "base_layering_horizon" : None, "horizon_type" : "conformable", "fault_lines" : find_object (name="top",
                      type="FaultLines"), "fault_lines_usage" : "do_not_use", "pinch_out_polygon" : None, "pinch_out_type" : "up"}, {"horizon" : find_object (name="bot",
                      type="Horizon"), "zone" : "Zone2", "partition_type" : "proportional", "counts_step" : 10, "individual_minimum_zone_thickness" : 0, "source_marker" : None, "layering_horizon" : None, "base_layering_horizon" : None, "horizon_type" : "conformable", "fault_lines" : find_object (name="bot",
                      type="FaultLines"), "fault_lines_usage" : "do_not_use", "pinch_out_polygon" : None, "pinch_out_type" : "up"}],
                      faults=[{"use" : True, "fault" : find_object (name="Fault1",
                      type="Fault3d"), "structure" : True, "zigzag" : False, "no_displace" : False, "left_distance" : 1000, "right_distance" : 1000}, {"use" : True, "fault" : find_object (name="Fault2",
                      type="Fault3d"), "structure" : True, "zigzag" : False, "no_displace" : False, "left_distance" : 1000, "right_distance" : 1000}, {"use" : True, "fault" : find_object (name="Fault3",
                      type="Fault3d"), "structure" : True, "zigzag" : False, "no_displace" : False, "left_distance" : 1000, "right_distance" : 1000}, {"use" : True, "fault" : find_object (name="Fault4",
                      type="Fault3d"), "structure" : True, "zigzag" : False, "no_displace" : False, "left_distance" : 1000, "right_distance" : 1000}, {"use" : True, "fault" : find_object (name="Fault5",
                      type="Fault3d"), "structure" : True, "zigzag" : False, "no_displace" : False, "left_distance" : 1000, "right_distance" : 1000}, {"use" : True, "fault" : find_object (name="Fault6",
                      type="Fault3d"), "structure" : True, "zigzag" : False, "no_displace" : False, "left_distance" : 1000, "right_distance" : 1000}],
                      residual_maps=False,
                      residual_maps_suffix="_Discrepancy",
                      general_settings=True,
                      wells=find_object (name="Wells",
                      type="gt_wells_entity"),
                      trajectories=find_object (name="Trajectories",
                      type="Trajectories"),
                      use_minimum_thickness=False,
                      minimum_thickness=0.1,
                      use_start_from=False,
                      start_from="from_top",
                      use_minimum_zone_thickness=False,
                      minimum_zone_thickness=0.1,
                      use_segments=True,
                      segments=find_object (name="Segments",
                      type="Grid3dProperty"),
                      use_region=False,
                      region=find_object (name="float_line_1_top",
                      type="Curve3d"),
                      border_policy="do_not_use",
                      use_well_filter=False,
                      well_filter=find_object (name="Well Filter 1",
                      type="WellFilter"),
                      use_auto_filtration_radius=True,
                      filtration_radius=100,
                      grid_2d_gui_data=Grid2DAutodetector (disabled=True,
                      autodetect_angle=False,
                      autodetect_sample_object=find_object (name="Wells",
                      type="gt_wells_entity"),
                      margin=0),
                      grid_2d=Grid2D (step_x=200,
                      step_y=200,
                      area=Rectangle (origin_x=0,
                      origin_y=0,
                      size_x=36200,
                      size_y=56400,
                      angle=0)),
                      create_grid_from_faults=False,
                      transversal_faults=[],
                      longitudinal_faults=[],
                      advanced_settings=False,
                      horizon_interpolation_parameters=False,
                      algorithm="least_squares",
                      first_derivative_coefficient=0.3,
                      second_derivative_coefficient=0.1,
                      do_drag=False,
                      drag_iterations=10,
                      drag_coefficient=0.5,
                      convergent_refinement_ratio=1.5,
                      faults_interpolation_parameters=False,
                      uv_algorithm="least_squares",
                      uv_first_derivative_coefficient=0.3,
                      uv_second_derivative_coefficient=0.01,
                      uv_do_drag=False,
                      uv_drag_iterations=10,
                      uv_drag_coefficient=0.5,
                      uv_convergent_refinement_ratio=1.5,
                      linearity_level=0,
                      extend_faults_to_z_borders=True,
                      simplify_faults=False,
                      use_extended_segments=False,
                      extended_segments=find_object (name="Segments_ext",
                      type="Grid3dProperty"),
                      use_grid_trend=False,
                      smooth_radius=0)
                end_wf_item (index = 23)


            if False:
                begin_wf_item (index = 24)
                grid_3d_move (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      angle=30,
                      center_x=0,
                      center_y=0,
                      shift_x=0,
                      shift_y=0,
                      shift_z=0,
                      use_invert_i=False,
                      use_invert_j=False)
                end_wf_item (index = 24)


            if False:
                begin_wf_item (index = 25)
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="X_",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="zone_id",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="X",
                      variables=variables)
                end_wf_item (index = 25)


            if False:
                begin_wf_item (index = 26)
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="Y_",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="zone_id",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="Y",
                      variables=variables)
                end_wf_item (index = 26)


            if False:
                begin_wf_item (index = 27)
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="Z_",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="zone_id",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="Z",
                      variables=variables)
                end_wf_item (index = 27)


            if False:
                begin_wf_item (index = 28, is_custom_code = True)
                g = get_grid_by_name(name='test_grid')
                property_x_ = g.get_property_by_name(name='X_')
                property_y_ = g.get_property_by_name(name='Y_')
                property_z_ = g.get_property_by_name(name='Z_')

                x = property_x_.get_np_array()
                y = property_y_.get_np_array()
                z = property_z_.get_np_array()

                coord = np.zeros((x.size, 3))
                index = 0
                for k in range(x.shape[2]):
                	print(f"Чтение слоя {k+1} из {x.shape[2]}")
                	for j in range(x.shape[1]):
                		for i in range(x.shape[0]):
                			coord[index, :] = np.array([x[i, j, k], y[i, j, k], z[i, j, k]])
                			index += 1

                file_path = r'D:\14. Кейсы\005. IRM\models\Example\02_oil\coord_directory\coords.INC'
                with open(file_path, 'w') as f:
                	f.write(f"{x.shape[0]} {x.shape[1]} {x.shape[2]}\n")
                	for row in coord:
                		f.write(' '.join(map(str, row)) + '\n')
                		
                file_path = r'D:\14. Кейсы\005. IRM\models\Example\01_gas\coord_directory\coords.INC'
                with open(file_path, 'w') as f:
                	f.write(f"{x.shape[0]} {x.shape[1]} {x.shape[2]}\n")
                	for row in coord:
                		f.write(' '.join(map(str, row)) + '\n')
                end_wf_item (index = 28)



        end_wf_item (index = 21)


    if False:
        begin_wf_item (index = 30)
        map_2d_import_user_image (file_name="map/ЯНАО.png",
              user_image_2d_data=find_object (name="map",
              type="User2DImages"),
              bind_mode_method="by_project",
              top_left_x=0,
              top_left_y=1000,
              bot_left_x=0,
              bot_left_y=0,
              bot_right_x=1000,
              bot_right_y=0,
              origin_x=0,
              origin_y=0,
              side_width=1000,
              side_height=1000)
        end_wf_item (index = 30)


    if False:
        begin_wf_item (index = 31, name = "Базовое свойство")
        workflow_folder ()
        if True:
            pass



            if False:
                begin_wf_item (index = 32, name = "sceleton")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="sceleton",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="0",
                      variables=variables)
                end_wf_item (index = 32)


            if False:
                begin_wf_item (index = 33, is_custom_code = True, name = "Filling sceleton")
                def generate_gaussian_peaks_V_1d(nz, num_peaks=8, zero=10):
                	''' Определение вертикального тренда скелетона с использованием линейной интерполяции '''
                	z_values = np.arange(nz)
                	random_values = np.zeros(nz)
                	
                	random_values[0] = 0.7
                	random_values[-1] = 0.5
                	
                	selected_indices = np.random.choice(np.arange(1, nz - 1), num_peaks - 2, replace=False)
                	selected_indices = np.sort(selected_indices)

                	selected_indices = np.insert(selected_indices, 0, 0)
                	selected_indices = np.append(selected_indices, nz - 1)

                	for idx in selected_indices:
                		if idx not in [0, nz - 1]:
                			random_values[idx] = np.random.rand()

                	f = interp1d(z_values[selected_indices], random_values[selected_indices], kind='linear', fill_value="extrapolate")
                	interpolated_values = f(z_values)

                	num_zeros = int(nz * zero / 100)
                	zero_indices = np.random.choice(nz, num_zeros, replace=False)
                	interpolated_values[zero_indices] = 0

                	return interpolated_values
                	
                def generate_gaussian_peaks_G_2d(nx, ny):
                	''' Определение горизонтального тренда скелетона '''
                	num_walks = 100										# количество случайных блужданий
                	walk_lengths = (500, 700)					# длина каждого блуждания
                	blur_sigma = 2.0										# сигма для гауссовского размытия


                	cave_array = np.zeros((nx, ny))
                	for _ in range(num_walks):
                		x, y = np.random.randint(0, nx), np.random.randint(0, ny)
                		for _ in range(np.random.randint(walk_lengths[0], walk_lengths[1])):
                			cave_array[x, y] = 1
                			dx, dy = np.random.normal(0, np.random.uniform(1, 2), 2)
                			x = round(np.clip(x + dx, 0, nx - 1))
                			y = round(np.clip(y + dy, 0, ny - 1))
                	cave_array = gaussian_filter(cave_array, sigma=blur_sigma)
                	cave_array = (cave_array - np.min(cave_array)) / (np.max(cave_array) - np.min(cave_array))
                	return cave_array

                def generate_property_in_blocks(data):
                	labels = np.unique(data)
                	medians = np.zeros(labels.shape)
                	for i, label in enumerate(labels):
                		medians[i] = np.random.uniform(0.5, 1.5)
                	return labels, medians

                g = get_grid_by_name(name='test_grid')
                property_sceleton = g.get_property_by_name(name='sceleton')
                property_bloks = g.get_property_by_name(name='Segments')
                data = property_sceleton.get_np_array()
                data_bloks = property_bloks.get_np_array()
                labels, medians = generate_property_in_blocks(data_bloks)

                new_data = np.zeros(data.shape)
                i, j, k = data.shape

                z_property = generate_gaussian_peaks_V_1d(k)
                xy = generate_gaussian_peaks_G_2d(i, j)

                for nz, trend_nz in enumerate(z_property):
                	print(f'z = {nz}, status = {trend_nz}')
                	new_data[:, :, nz] =  xy * trend_nz + np.clip(np.random.normal(0, np.random.uniform(0.5, 1.5), (i, j)), 0, 0.05)

                for label, median in zip(labels, medians):
                	print(f'label = {nz}, median = {median}')
                	mask = data_bloks == label
                	new_data[mask] = new_data[mask] * median
                	
                new_data  = (new_data - np.min(new_data)) / (np.max(new_data) - np.min(new_data))
                property_sceleton.set_np_array(values=new_data)




                end_wf_item (index = 33)



        end_wf_item (index = 31)


    if False:
        begin_wf_item (index = 35, name = "Пористость", collapsed = True)
        workflow_folder ()
        if True:
            pass



            if False:
                begin_wf_item (index = 36, name = "PORO")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="PORO",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="0",
                      variables=variables)
                end_wf_item (index = 36)


            if False:
                begin_wf_item (index = 37, is_custom_code = True, name = "Filling PORO")
                def polynomial_approximation(x, y, degree=2):
                	''' Полиномиальная аппроксимация '''
                	coefficients = np.polyfit(x, y, degree)
                	polynomial = np.poly1d(coefficients)
                	return polynomial

                def map_values(data, x_points, y_points, degree=2):
                	''' Маппинг значений по полиномиальной аппроксимации '''
                	polynomial = polynomial_approximation(x_points, y_points, degree)
                	mapped_values = polynomial(data)
                	cave_array =  mapped_values * (100 - np.random.normal(0, 5, mapped_values.shape))/100 + np.random.uniform(-0.02, 0.02, mapped_values.shape)
                	cave_array = gaussian_filter(cave_array, sigma=0.8)
                	return cave_array

                new_min = 0.02
                new_max = 0.3
                new_point = PORO

                g = get_grid_by_name(name='test_grid')
                property_sceleton = g.get_property_by_name(name='sceleton')
                sceleton = property_sceleton.get_np_array()

                x_points = [np.min(sceleton), np.median(sceleton), np.max(sceleton)]
                y_points = [new_min, new_point, new_max]
                props = map_values(sceleton, x_points, y_points)

                property_props = g.get_property_by_name(name='PORO')
                property_props.set_np_array(values=props)
                end_wf_item (index = 37)



        end_wf_item (index = 35)


    if False:
        begin_wf_item (index = 39, name = "Активные ячейки", collapsed = True)
        workflow_folder ()
        if True:
            pass



            if False:
                begin_wf_item (index = 40, name = "ACTNUM")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="ACTNUM",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="0",
                      variables=variables)
                end_wf_item (index = 40)


            if False:
                begin_wf_item (index = 41, is_custom_code = True, name = "Filling Actnum")
                g = get_grid_by_name(name='test_grid')
                property_trend = g.get_property_by_name(name='ACTNUM')
                property_props = g.get_property_by_name(name='PORO')

                porosity = property_props.get_np_array()
                active = np.where(0.08 <= porosity, 1, 0)
                actnum = np.zeros(active.shape)
                for k in range(porosity.shape[-1]):
                	print(f'Actnum layer = {k+1} in {porosity.shape[-1]}')
                	actnum[:, :, k] = median_filter(active[:, :, k], size=15)
                property_trend.set_np_array(values=actnum)

                end_wf_item (index = 41)



        end_wf_item (index = 39)


    if False:
        begin_wf_item (index = 43, name = "Проницаемость", collapsed = True)
        workflow_folder ()
        if True:
            pass



            if False:
                begin_wf_item (index = 44, name = "PERM")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="PERMX",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="0",
                      variables=variables)
                end_wf_item (index = 44)


            if False:
                begin_wf_item (index = 45, is_custom_code = True, name = "Filling PERM")
                g = get_grid_by_name(name='test_grid')
                property_trend = g.get_property_by_name(name='PORO')
                trend = property_trend.get_np_array()

                a = 4204.9
                b = 2.6007


                props = np.random.uniform(0.95, 1.05, trend.shape) * a * np.power(trend, b)
                props +=np.random.normal(0, np.mean(props)/ 10, props.shape)
                props = gaussian_filter(props, sigma=0.8)
                props = np.clip(props, 0.01, np.max(props))

                print(f'PERM, min={np.min(props)}, max={np.min(props)}, P90={np.percentile(props, 10)}, P50={np.percentile(props, 50)}, P10={np.percentile(props, 90)}')

                property_props = g.get_property_by_name(name='PERMX')
                property_props.set_np_array(values=props)
                end_wf_item (index = 45)


            if False:
                begin_wf_item (index = 46, name = "PERM")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="PERMY",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="PERMX",
                      variables=variables)
                end_wf_item (index = 46)



        end_wf_item (index = 43)


    if False:
        begin_wf_item (index = 48, name = "Песчанистость", collapsed = True)
        workflow_folder ()
        if True:
            pass



            if False:
                begin_wf_item (index = 49, name = "NTG")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="NTG",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="0",
                      variables=variables)
                end_wf_item (index = 49)


            if False:
                begin_wf_item (index = 50, is_custom_code = True, name = "Filling NTG")
                g = get_grid_by_name(name='test_grid')
                property_trend = g.get_property_by_name(name='PERMX')
                trend = property_trend.get_np_array()

                a = 0.1814
                b = -0.0593

                props = a * np.log(trend) + b
                props +=np.random.normal(0, np.mean(props)/ 20, props.shape)
                props = np.clip(props, 0, 1)

                print(f'NTG, min={np.min(props)}, max={np.min(props)}, P90={np.percentile(props, 10)}, P50={np.percentile(props, 50)}, P10={np.percentile(props, 90)}')

                property_props = g.get_property_by_name(name='NTG')
                property_props.set_np_array(values=props)
                end_wf_item (index = 50)



        end_wf_item (index = 48)


    if False:
        begin_wf_item (index = 52, name = "Анизотропия по вертикали", collapsed = True)
        workflow_folder ()
        if True:
            pass



            if False:
                begin_wf_item (index = 53, name = "Kaniz")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="Kaniz",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="0",
                      variables=variables)
                end_wf_item (index = 53)


            if False:
                begin_wf_item (index = 54, is_custom_code = True, name = "Filling Kaniz")
                g = get_grid_by_name(name='test_grid')
                property_trend = g.get_property_by_name(name='NTG')
                trend = property_trend.get_np_array()

                a = 0.1213
                b = 2.0296

                props = a * np.power(trend, b)

                props +=np.random.normal(0, np.mean(props)/ 10, props.shape)
                props = np.clip(props, 0.001, np.max(props))

                print(f'Kaniz, min={np.min(props)}, max={np.min(props)}, P90={np.percentile(props, 10)}, P50={np.percentile(props, 50)}, P10={np.percentile(props, 90)}')

                property_props = g.get_property_by_name(name='Kaniz')
                property_props.set_np_array(values=props)
                end_wf_item (index = 54)


            if False:
                begin_wf_item (index = 55, name = "PERMZ")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="PERMZ",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="PERMX * Kaniz",
                      variables=variables)
                end_wf_item (index = 55)



        end_wf_item (index = 52)


    if False:
        begin_wf_item (index = 57, name = "Остаточная водонасыщенность", collapsed = True)
        workflow_folder ()
        if True:
            pass



            if False:
                begin_wf_item (index = 58, name = "SWL")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="SWL",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="0",
                      variables=variables)
                end_wf_item (index = 58)


            if False:
                begin_wf_item (index = 59, is_custom_code = True, name = "Filling SWL")
                g = get_grid_by_name(name='test_grid')
                property_trend = g.get_property_by_name(name='PERMX')
                trend = property_trend.get_np_array()

                a = 0.9028
                b =-0.235

                props = a * np.power(trend, b)

                props +=np.random.normal(0, np.mean(props)/ 20, props.shape)
                props = np.clip(props, 0.001, 1)

                print(f'SWL, min={np.min(props)}, max={np.min(props)}, P90={np.percentile(props, 10)}, P50={np.percentile(props, 50)}, P10={np.percentile(props, 90)}')

                property_props = g.get_property_by_name(name='SWL')
                property_props.set_np_array(values=props)
                end_wf_item (index = 59)



        end_wf_item (index = 57)


    if False:
        begin_wf_item (index = 61, name = "PVT", collapsed = True)
        workflow_folder ()
        if True:
            pass



            if False:
                begin_wf_item (index = 62, is_custom_code = True, name = "save_variables")
                def name_variable(name: str, value: float, path: str = r'D:\14. Кейсы\005. IRM\models\Variables'):
                	with open(f"{path}\{name}.txt", 'w') as file:
                		file.write(f"{value}\n")
                		
                name_variable('PRESSURE', PRESSURE)
                name_variable('TEMPERATURE', TEMPERATURE)
                name_variable('DENSITY_OIL', DENSITY_OIL)
                name_variable('DENSITY_WATER', DENSITY_WATER)
                name_variable('DENSITY_GAS', DENSITY_GAS)
                end_wf_item (index = 62)


            if False:
                begin_wf_item (index = 63, collapsed = True)
                run_project_workflow (project_type = "pvt_project",
                      project_name = "PVT Data",
                      workflow = "PVT",
                      variable_types = {},
                      variables_object = {

                })
                end_wf_item (index = 63)



        end_wf_item (index = 61)


    if False:
        begin_wf_item (index = 65, name = "Насыщенности", collapsed = True)
        workflow_folder ()
        if True:
            pass



            if False:
                begin_wf_item (index = 66, name = "Характер насыщения")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="GasOilWater_zone",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="0",
                      variables=variables)
                end_wf_item (index = 66)


            if False:
                begin_wf_item (index = 67, name = "Глубина")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="Depth",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="Z",
                      variables=variables)
                end_wf_item (index = 67)


            if False:
                begin_wf_item (index = 68, is_custom_code = True, name = "Определение плотности газа нефти в пластовых условиях")
                # Плотность газа в пластовых условиях
                def read_pvdg_data(file_path):
                	pressure = []
                	fvf = []
                	with open(file_path, 'r') as file:
                		recording = False
                		for line in file:
                			if 'PVDG' in line:
                				recording = True
                				continue
                			if recording:
                				if line.startswith('--'):
                					continue
                				if line.strip().endswith('/'):
                					break
                				parts = line.split()
                				print(float(parts[1]), float(parts[0]), parts)
                				if len(parts) >= 2:
                					pressure.append(float(parts[0]))
                					fvf.append(float(parts[1]))
                	return np.array(pressure), np.array(fvf)

                def read_pvto_data(file_path):
                	gor = []
                	pressure_mass, fvf_mass, recording,  = [], [], False
                	status, pressure, fvf = False, [], []
                	with open(file_path, 'r') as file:
                		for line in file:
                			if 'PVTO' in line:
                				recording = True
                				continue
                			if line.split() == ['/']:
                				break
                			if recording:
                				if line.startswith('--'):
                					continue
                				patrs = line.split()
                				if patrs[-1] == '/':
                					status = True
                				if len(patrs) == 4 and patrs[-1] != '/':
                					gor.append(float(patrs[0]))
                					pressure.append(float(patrs[1]))
                					fvf.append(float(patrs[2]))
                				else:
                					pressure.append(float(patrs[0]))
                					fvf.append(float(patrs[1]))
                				if status:
                					pressure_mass.append(pressure)
                					fvf_mass.append(fvf)
                					fvf, pressure = [], []
                					status = False
                	return gor[-1], np.array(pressure_mass[-15]), np.array(fvf_mass[-15])	#Не корректно

                def get_fvf(p, pressure, fvf, name):
                	print(p, len(pressure), len(fvf), name)
                	print(pressure, fvf)
                	interp_func = interp1d(pressure, fvf, kind='linear', fill_value="extrapolate")
                	return interp_func(p)

                pressure_gas, fvf_gas = read_pvdg_data(r'D:\14. Кейсы\005. IRM\models\PVT\Вариант 1.inc')
                gor, pressure_oil, fvf_oil = read_pvto_data(r'D:\14. Кейсы\005. IRM\models\PVT\Вариант 1.inc')

                density_gas_pl = DENSITY_GAS / get_fvf(PRESSURE, pressure_gas, fvf_gas, 'gas')
                density_oil_pl = DENSITY_OIL / get_fvf(PRESSURE, pressure_oil, fvf_oil, 'oil')

                '''
                def name_variable(name: str, value: float, path: str = r'D:\14. Кейсы\005. IRM\models\Variables'):
                	with open(f"{path}\{name}.txt", 'w') as file:
                		file.write(f"{value}\n")

                name_variable('density_gas_pl', density_gas_pl)
                name_variable('density_oil_pl', density_oil_pl)
                '''
                end_wf_item (index = 68)


            if False:
                begin_wf_item (index = 69, is_custom_code = True, name = "Filling Zone")
                g = get_grid_by_name(name='test_grid')
                property_trend = g.get_property_by_name(name='Segments')
                property_props = g.get_property_by_name(name='GasOilWater_zone')
                property_depth = g.get_property_by_name(name='Depth')

                trends = property_trend.get_np_array()
                ogw = property_props.get_np_array()
                depth = property_depth.get_np_array()

                ulabels = list(np.unique(trends))
                property_props = g.get_property_by_name(name='GasOilWater_zone')
                ogw = property_props.get_np_array()

                print(f'DEPTH, min={np.min(depth)}, max={np.min(depth)}, P90={np.percentile(depth, 10)}, P50={np.percentile(depth, 50)}, P10={np.percentile(depth, 90)}')

                print(f'GLOBAL_VNK = {GLOBAL_VNK}; GLOBAL_GNK = {GLOBAL_GNK }')

                dict_GNK, dict_VNK = {}, {}

                for ulabel in ulabels:
                	new_GNK = np.random.uniform(-25, 5) + GLOBAL_GNK
                	new_VNK = np.random.uniform(-5, 25) + GLOBAL_VNK

                	dict_GNK[ulabel] = new_GNK
                	dict_VNK[ulabel] = new_VNK

                	if new_GNK < new_VNK:
                		print(f'zone OGW - {ulabel} in {len(ulabels)} create:GNK={round(new_GNK)}, VNK={round(new_VNK)}')
                		mask = trends == ulabel
                		ogw[mask & (depth > new_VNK)] = 0
                		ogw[mask & (depth < new_GNK)] = 2
                		ogw[mask & (new_VNK >= depth) & (depth >= new_GNK)] = 1
                	else:
                		print(f'zone GW- {ulabel} in {len(ulabels)} create:GNK={round(new_GNK)}, VNK={round(new_VNK)}')
                		mask = trends == ulabel
                		ogw[mask & (depth >= new_GNK)] = 0
                		ogw[mask & (depth < new_GNK)] = 2


                property_props.set_np_array(values=ogw)




                end_wf_item (index = 69)


            if False:
                begin_wf_item (index = 70, name = "Pc вода-нефть")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="pc_wo",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="0\n",
                      variables=variables)
                end_wf_item (index = 70)


            if False:
                begin_wf_item (index = 71, name = "Pc вода-газ")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="pc_wg",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="0",
                      variables=variables)
                end_wf_item (index = 71)


            if False:
                begin_wf_item (index = 72, is_custom_code = True, name = "Filling Pc")
                property_pc_wo = g.get_property_by_name(name='pc_wo')
                property_pc_wg = g.get_property_by_name(name='pc_wg')
                property_zone = g.get_property_by_name(name='GasOilWater_zone')
                property_depth = g.get_property_by_name(name='Depth')
                property_trend = g.get_property_by_name(name='Segments')

                pc_wo = property_pc_wo.get_np_array()
                pc_wg = property_pc_wg.get_np_array()
                depth = property_depth.get_np_array()
                ogw_zone = property_zone.get_np_array()
                blocks = property_trend.get_np_array()

                ulabels = list(np.unique(blocks))
                for ulabel in ulabels:
                	print(f'zone OGW - {ulabel}')
                	mask = blocks == ulabel
                	# нефть
                	pc_wg[mask & (ogw_zone == 1)] = (DENSITY_WATER - density_gas_pl) * (dict_VNK[ulabel] - depth[mask & (ogw_zone == 1)]) * 9.81 * 0.00001
                	pc_wo[mask & (ogw_zone == 1)] = (DENSITY_WATER - density_oil_pl) * (dict_VNK[ulabel] - depth[mask & (ogw_zone == 1)]) * 9.81 * 0.00001
                	# газ
                	pc_wg[mask & (ogw_zone == 2)] = (DENSITY_WATER - density_gas_pl) * (dict_VNK[ulabel] - depth[mask & (ogw_zone == 2)]) * 9.81 * 0.00001
                	
                property_pc_wo.set_np_array(values=pc_wo)
                property_pc_wg.set_np_array(values=pc_wg)
                end_wf_item (index = 72)


            if False:
                begin_wf_item (index = 73, name = "J-функция вода-нефть")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="Jf_wo",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="(pc_wo / 73 ) *sqrt (PERMX/PORO)*3.1415",
                      variables=variables)
                end_wf_item (index = 73)


            if False:
                begin_wf_item (index = 74, name = "J-функция вода-газ")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="Jf_wg",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="(pc_wg / 16 ) *sqrt (PERMX/PORO)*3.1415",
                      variables=variables)
                end_wf_item (index = 74)


            if False:
                begin_wf_item (index = 75, name = "Sw norm")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="Sw_WO_norm",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="if(Jf_wo==0,1,min(pow((Jf_wo/0.05), (1/(-1.5))),1))",
                      variables=variables)
                end_wf_item (index = 75)


            if False:
                begin_wf_item (index = 76, name = "Sw norm")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="Sw_WG_norm",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="if(Jf_wg==0,1,min(pow((Jf_wg/0.05), (1/(-1.5))),1))",
                      variables=variables)
                end_wf_item (index = 76)


            if False:
                begin_wf_item (index = 77, name = "Sw WO")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="Sw_WO",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="SWL + (1 - SWL)*Sw_WO_norm",
                      variables=variables)
                end_wf_item (index = 77)


            if False:
                begin_wf_item (index = 78, name = "Sw WG")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="Sw_WG",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="SWL + (1 - SWL)*Sw_WG_norm",
                      variables=variables)
                end_wf_item (index = 78)


            if False:
                begin_wf_item (index = 79, name = "Sw")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="Sw_init",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="if(GasOilWater_zone == 1, Sw_WO, Sw_WG)",
                      variables=variables)
                end_wf_item (index = 79)


            if False:
                begin_wf_item (index = 80, name = "Sg")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="Sg_init",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="if(GasOilWater_zone == 2, 1 - Sw_init, 0)",
                      variables=variables)
                end_wf_item (index = 80)


            if False:
                begin_wf_item (index = 81, name = "So")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="So__init",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="if(GasOilWater_zone == 1, 1 - Sw_init, 0)",
                      variables=variables)
                end_wf_item (index = 81)


            if False:
                begin_wf_item (index = 82, name = "So")
                grid_property_calculator (mesh=find_object (name="test_grid",
                      type="Grid3d"),
                      result_grid_property=find_object (name="test",
                      type="Grid3dProperty"),
                      use_filter=False,
                      user_cut_for_filter=find_object (name="Property1",
                      type="Grid3dProperty"),
                      filter_comparator=Comparator (rule="not_equals",
                      value=0),
                      formula="if(Sw_init + Sg_init + So__init > 1, 1, 0)",
                      variables=variables)
                end_wf_item (index = 82)



        end_wf_item (index = 65)


    if False:
        begin_wf_item (index = 84, name = "ОФП", collapsed = True)
        workflow_folder ()
        if True:
            pass



            if False:
                begin_wf_item (index = 85)
                rp_create_by_corey (correlation_type="Corey",
                      rp_phases="oil_and_gas",
                      clear_tables=True,
                      wo_params_tables=[{"table_name" : "Table", "S_PL" : 0.3, "S_PU" : 0.9, "S_PCR" : 0.33, "S_OPCR" : 0.43, "k_rOLP" : 1, "k_rORP" : 1, "k_rPR" : 0.2, "k_rPU" : 1, "p_cOP" : 0.03, "N_OP" : 1.5, "N_P" : 2.5, "N_pcap" : 1.5, "S_OPL" : 0, "S_pc" : -1, "T_P" : 0, "E_P" : 0, "T_OP" : 0, "E_OP" : 0}],
                      go_params_tables=[{"table_name" : "Table", "S_PL" : 0.1, "S_PU" : 0.7, "S_PCR" : 0.11, "S_OPCR" : 0.35, "k_rOLP" : 1, "k_rORP" : 1, "k_rPR" : 0.6, "k_rPU" : 1, "p_cOP" : 0, "N_OP" : 1.5, "N_P" : 2.5, "N_pcap" : 1.5, "S_OPL" : 0, "S_pc" : -1, "T_P" : 0, "E_P" : 0, "T_OP" : 0, "E_OP" : 0}],
                      wg_params_tables=[{"table_name" : "Table", "S_PL" : 0.05, "S_PU" : 0.95, "S_PCR" : 0.15, "S_OPCR" : 0.3, "k_rOLP" : 0.9, "k_rORP" : 0.7, "k_rPR" : 0.4, "k_rPU" : 0.55, "p_cOP" : 0.11, "N_OP" : 2, "N_P" : 2, "N_pcap" : 2, "S_OPL" : 0, "S_pc" : 0.2, "T_P" : 0, "E_P" : 0, "T_OP" : 0, "E_OP" : 0}])
                end_wf_item (index = 85)


            if False:
                begin_wf_item (index = 86, name = "Вода - нефть")
                workflow_folder ()
                if True:
                    pass



                    if False:
                        begin_wf_item (index = 87, name = "SWCR")
                        grid_property_calculator (mesh=find_object (name="test_grid",
                              type="Grid3d"),
                              result_grid_property=find_object (name="SWCR",
                              type="Grid3dProperty"),
                              use_filter=False,
                              user_cut_for_filter=find_object (name="zone_id",
                              type="Grid3dProperty"),
                              filter_comparator=Comparator (rule="not_equals",
                              value=0),
                              formula="SWL + 0.04",
                              variables=variables)
                        end_wf_item (index = 87)


                    if False:
                        begin_wf_item (index = 88, name = "SOWCR")
                        grid_property_calculator (mesh=find_object (name="test_grid",
                              type="Grid3d"),
                              result_grid_property=find_object (name="SOWCR",
                              type="Grid3dProperty"),
                              use_filter=False,
                              user_cut_for_filter=find_object (name="zone_id",
                              type="Grid3dProperty"),
                              filter_comparator=Comparator (rule="not_equals",
                              value=0),
                              formula="if(1.49*PORO+8.25/100+SWCR>0.99,0.99-SWCR,1.49*PORO+8.25/100)\n",
                              variables=variables)
                        end_wf_item (index = 88)


                    if False:
                        begin_wf_item (index = 89, name = "KRW")
                        grid_property_calculator (mesh=find_object (name="test_grid",
                              type="Grid3d"),
                              result_grid_property=find_object (name="KRW",
                              type="Grid3dProperty"),
                              use_filter=False,
                              user_cut_for_filter=find_object (name="zone_id",
                              type="Grid3dProperty"),
                              filter_comparator=Comparator (rule="not_equals",
                              value=0),
                              formula="if(0.0008437*EXP(3.9609713*PERMX)*3>0.8, 0.8, 0.0008437*EXP(3.9609713*PERMX)*3)",
                              variables=variables)
                        end_wf_item (index = 89)


                    if False:
                        begin_wf_item (index = 90, name = "KRO")
                        grid_property_calculator (mesh=find_object (name="test_grid",
                              type="Grid3d"),
                              result_grid_property=find_object (name="KRO",
                              type="Grid3dProperty"),
                              use_filter=False,
                              user_cut_for_filter=find_object (name="zone_id",
                              type="Grid3dProperty"),
                              filter_comparator=Comparator (rule="not_equals",
                              value=0),
                              formula="if(0.010159*EXP(3.140385*PERMX)*3>0.8,0.8,0.010159*EXP(3.140385*PERMX)*3)",
                              variables=variables)
                        end_wf_item (index = 90)


                    if False:
                        begin_wf_item (index = 91, name = "KRWR")
                        grid_property_calculator (mesh=find_object (name="test_grid",
                              type="Grid3d"),
                              result_grid_property=find_object (name="KRWR",
                              type="Grid3dProperty"),
                              use_filter=False,
                              user_cut_for_filter=find_object (name="zone_id",
                              type="Grid3dProperty"),
                              filter_comparator=Comparator (rule="not_equals",
                              value=0),
                              formula="if(0.0000000488*EXP(39.5919260694*(SOWCR+0.02))*3>KRW,KRW,0.0000000488*EXP(39.5919260694*(SOWCR+0.02))*3)",
                              variables=variables)
                        end_wf_item (index = 91)


                    if False:
                        begin_wf_item (index = 92, name = "KRORW")
                        grid_property_calculator (mesh=find_object (name="test_grid",
                              type="Grid3d"),
                              result_grid_property=find_object (name="KRORW",
                              type="Grid3dProperty"),
                              use_filter=False,
                              user_cut_for_filter=find_object (name="zone_id",
                              type="Grid3dProperty"),
                              filter_comparator=Comparator (rule="not_equals",
                              value=0),
                              formula="if(0.00000246*EXP(32.67630164*(SOWCR+0.02))*3>KRO,KRO,0.00000246*EXP(32.67630164*(SOWCR+0.02))*3)",
                              variables=variables)
                        end_wf_item (index = 92)



                end_wf_item (index = 86)



        end_wf_item (index = 84)


    if False:
        begin_wf_item (index = 95, name = "EQUIL", collapsed = True)
        workflow_folder ()
        if True:
            pass



            if False:
                begin_wf_item (index = 96, is_custom_code = True, name = "Генерация контактов")
                def write_equil(text_line: str, status: bool=False):
                	# Поменять путь
                	file_path = r"D:\14. Кейсы\005. IRM\models\Equil\EQUIL.inc"
                	mode = 'w' if status else 'a'
                	with open(file_path, mode) as file:
                		file.write(text_line + '\n')

                write_equil('EQLDIMS', status=True)
                write_equil(f'{int(len(list(dict_GNK.values())))}')
                write_equil('/')
                write_equil('')
                write_equil('EQUIL')
                for gnk_line, vnk_line in zip(list(dict_GNK.values()), list(dict_VNK.values())):
                	write_equil(f'{round(vnk_line)} {round(PRESSURE)} {round(vnk_line)} 0 {round(gnk_line)} 0 1 0 /')
                write_equil('/')
                write_equil('')
                write_equil('RSVD')
                for gnk_line in list(dict_GNK.values()):
                	write_equil(f'{round(gnk_line-10)} {round(gor)}')
                	write_equil(f'{round(gnk_line)} {round(gor)}')
                	write_equil('/')



                end_wf_item (index = 96)


            if False:
                begin_wf_item (index = 97, name = "EQUIL")
                simple_props_import_e100_format (prop_item_type="equil_v2",
                      region_count=7,
                      file_name="D:/14. Кейсы/005. IRM/models/Equil/EQUIL.inc",
                      units_system="METRIC",
                      clear_tables=True)
                end_wf_item (index = 97)


            if False:
                begin_wf_item (index = 98)
                adjust_props_rock (property_name="Сжимаемость породы 1",
                      property_folder="Сжимаемость породы",
                      values=[[arithmetic (expression=resolve_variables_in_string (string_with_variables="@PRESSURE@",
                      variables=variables),
                      variables=variables), 0.00005]],
                      mode="adjust")
                end_wf_item (index = 98)


            if False:
                begin_wf_item (index = 99, name = "RSVD")
                simple_props_import_e100_format (prop_item_type="rsvd",
                      region_count=7,
                      file_name="D:/14. Кейсы/005. IRM/models/Equil/EQUIL.inc",
                      units_system="METRIC",
                      clear_tables=True)
                end_wf_item (index = 99)



        end_wf_item (index = 95)


    if False:
        begin_wf_item (index = 101, name = "create GDM", collapsed = True)
        workflow_folder ()
        if True:
            pass



            if False:
                begin_wf_item (index = 102)
                runspec_mapping (cases=find_object (name="DynamicModel",
                      type="Model_ex"),
                      action="add",
                      autofill=True,
                      runspec_table=[{"hierarchy" : "main", "kw_group" : "Fluid Composition", "description" : "В расчёте есть вода", "kw" : "WATER", "buffer" : ""}, {"hierarchy" : "main", "kw_group" : "Fluid Composition", "description" : "В расчёте есть газ", "kw" : "GAS", "buffer" : ""}, {"hierarchy" : "main", "kw_group" : "Fluid Composition", "description" : "В расчёте есть растворенный в нефти газ", "kw" : "DISGAS", "buffer" : ""}, {"hierarchy" : "main", "kw_group" : "Dimensions", "description" : "Размерности таблиц равновесия", "kw" : "EQLDIMS", "buffer" : "/\n\n"}, {"hierarchy" : "main", "kw_group" : "Tuning", "description" : "Настройки чтения и расчёта модели", "kw" : "TNAVCTRL", "buffer" : "LONGNAMES YES  /\n E1_E3_BINARY_EXPORT ON  /\n /\n\n"}, {"hierarchy" : "main", "kw_group" : "Other", "description" : "Опции равновесия", "kw" : "EQLOPTS", "buffer" : "NONEQSAT /\n"}, {"hierarchy" : "main", "kw_group" : "Options", "description" : "Исп. масштабирование концевых точек", "kw" : "ENDSCALE", "buffer" : "2* 1 /\n\n"}, {"hierarchy" : "main", "kw_group" : "Fluid Composition", "description" : "Задаёт, что в расчёте есть нефть", "kw" : "OIL", "buffer" : ""}, {"hierarchy" : "main", "kw_group" : "Options", "description" : "Задаёт объединение выходных файлов", "kw" : "UNIFOUT", "buffer" : ""}],
                      variables=variables)
                end_wf_item (index = 102)


            if False:
                begin_wf_item (index = 103)
                static_mapping (cases=find_object (name="DynamicModel",
                      type="Model_ex"),
                      grid=find_object (name="test_grid",
                      type="Grid3d"),
                      set_grid=True,
                      action="replace",
                      static_table=[{"description" : "ID_KRORW", "keyword" : "KRORW", "component" : None, "property" : find_object (name="KRORW",
                      type="Grid3dProperty"), "constant" : None, "porosity" : "matrix"}, {"description" : "ID_EQLNUM", "keyword" : "EQLNUM", "component" : None, "property" : find_object (name="Segments",
                      type="Grid3dProperty"), "constant" : None, "porosity" : "matrix"}, {"description" : "ID_PERMX", "keyword" : "PERMX", "component" : None, "property" : find_object (name="PERMX",
                      type="Grid3dProperty"), "constant" : None, "porosity" : "matrix"}, {"description" : "ID_KRO", "keyword" : "KRO", "component" : None, "property" : find_object (name="KRO",
                      type="Grid3dProperty"), "constant" : None, "porosity" : "matrix"}, {"description" : "ID_SOIL", "keyword" : "SOIL", "component" : None, "property" : find_object (name="So__init",
                      type="Grid3dProperty"), "constant" : None, "porosity" : "matrix"}, {"description" : "ID_SATNUM", "keyword" : "SATNUM", "component" : None, "property" : find_object (name="Segments",
                      type="Grid3dProperty"), "constant" : None, "porosity" : "matrix"}, {"description" : "ID_PORO", "keyword" : "PORO", "component" : None, "property" : find_object (name="PORO",
                      type="Grid3dProperty"), "constant" : None, "porosity" : "matrix"}, {"description" : "ID_SGAS", "keyword" : "SGAS", "component" : None, "property" : find_object (name="Sg_init",
                      type="Grid3dProperty"), "constant" : None, "porosity" : "matrix"}, {"description" : "ID_PERMZ", "keyword" : "PERMZ", "component" : None, "property" : find_object (name="PERMZ",
                      type="Grid3dProperty"), "constant" : None, "porosity" : "matrix"}, {"description" : "ID_SOWCR", "keyword" : "SOWCR", "component" : None, "property" : find_object (name="SOWCR",
                      type="Grid3dProperty"), "constant" : None, "porosity" : "matrix"}, {"description" : "ID_NTG", "keyword" : "NTG", "component" : None, "property" : find_object (name="NTG",
                      type="Grid3dProperty"), "constant" : None, "porosity" : "matrix"}, {"description" : "ID_SWCR", "keyword" : "SWCR", "component" : None, "property" : find_object (name="SWCR",
                      type="Grid3dProperty"), "constant" : None, "porosity" : "matrix"}, {"description" : "ID_SWAT", "keyword" : "SWAT", "component" : None, "property" : find_object (name="Sw_init",
                      type="Grid3dProperty"), "constant" : None, "porosity" : "matrix"}, {"description" : "ID_KRWR", "keyword" : "KRWR", "component" : None, "property" : find_object (name="KRWR",
                      type="Grid3dProperty"), "constant" : None, "porosity" : "matrix"}, {"description" : "ID_PERMY", "keyword" : "PERMY", "component" : None, "property" : find_object (name="PERMY",
                      type="Grid3dProperty"), "constant" : None, "porosity" : "matrix"}, {"description" : "ID_KRW", "keyword" : "KRW", "component" : None, "property" : find_object (name="KRW",
                      type="Grid3dProperty"), "constant" : None, "porosity" : "matrix"}, {"description" : "ID_ACTNUM", "keyword" : "ACTNUM", "component" : None, "property" : find_object (name="ACTNUM",
                      type="Grid3dProperty"), "constant" : None, "porosity" : "matrix"}, {"description" : "ID_SWL", "keyword" : "SWL", "component" : None, "property" : find_object (name="SWL",
                      type="Grid3dProperty"), "constant" : None, "porosity" : "matrix"}, {"description" : "ID_FIPXXX", "keyword" : "FIPNUM", "component" : None, "property" : find_object (name="Segments",
                      type="Grid3dProperty"), "constant" : None, "porosity" : "matrix"}])
                end_wf_item (index = 103)


            if False:
                begin_wf_item (index = 104)
                fluids_mapping (cases=find_object (name="DynamicModel",
                      type="Model_ex"),
                      group="PVT",
                      type="blackoil pvt",
                      action="replace",
                      fluids_table=[{"region" : 1, "property" : "Вариант 1 (PVT Data)"}])
                end_wf_item (index = 104)


            if False:
                begin_wf_item (index = 105)
                fluids_mapping (cases=find_object (name="DynamicModel",
                      type="Model_ex"),
                      group="Rock",
                      type="rock",
                      action="replace",
                      fluids_table=[{"region" : 1, "property" : "Сжимаемость породы 1"}])
                end_wf_item (index = 105)


            if False:
                begin_wf_item (index = 106)
                fluids_mapping (cases=find_object (name="DynamicModel",
                      type="Model_ex"),
                      group="Initial",
                      type="equil_v2",
                      action="replace",
                      fluids_table=[{"region" : 1, "property" : "EQUIL.inc 1 1"}, {"region" : 2, "property" : "EQUIL.inc 2 1"}, {"region" : 3, "property" : "EQUIL.inc 3 1"}, {"region" : 4, "property" : "EQUIL.inc 4 1"}, {"region" : 5, "property" : "EQUIL.inc 5 1"}, {"region" : 6, "property" : "EQUIL.inc 6 1"}, {"region" : 7, "property" : "EQUIL.inc 7 1"}])
                end_wf_item (index = 106)


            if False:
                begin_wf_item (index = 107)
                fluids_mapping (cases=find_object (name="DynamicModel",
                      type="Model_ex"),
                      group="Initial",
                      type="rsvd",
                      action="replace",
                      fluids_table=[{"region" : 1, "property" : "EQUIL.inc 1 2"}, {"region" : 2, "property" : "EQUIL.inc 2 2"}, {"region" : 3, "property" : "EQUIL.inc 3 2"}, {"region" : 4, "property" : "EQUIL.inc 4 2"}, {"region" : 5, "property" : "EQUIL.inc 5 2"}, {"region" : 6, "property" : "EQUIL.inc 6 2"}, {"region" : 7, "property" : "EQUIL.inc 7 2"}])
                end_wf_item (index = 107)


            if False:
                begin_wf_item (index = 108)
                fluids_mapping (cases=find_object (name="DynamicModel",
                      type="Model_ex"),
                      group="KRP",
                      type="rp_project",
                      action="replace",
                      fluids_table=[{"region" : 1, "property" : "Table (Проект ОФП)"}, {"region" : 2, "property" : "Table (Проект ОФП)"}, {"region" : 3, "property" : "Table (Проект ОФП)"}, {"region" : 4, "property" : "Table (Проект ОФП)"}, {"region" : 5, "property" : "Table (Проект ОФП)"}, {"region" : 6, "property" : "Table (Проект ОФП)"}, {"region" : 7, "property" : "Table (Проект ОФП)"}])
                end_wf_item (index = 108)


            if False:
                begin_wf_item (index = 109)
                open_or_reload_dynamic_model (use_model=False,
                      model=find_object (name="DynamicModel",
                      type="Model_ex"),
                      result_name="result")
                end_wf_item (index = 109)


            begin_wf_item (index = 110, name = "Посянения к формату выгружаемых кубов")
            comment_text ("""
Кубы свойст нужно сохранять в формате .INC
""")
            end_wf_item (index = 110)


            if False:
                begin_wf_item (index = 111, name = "Нефть")
                workflow_folder ()
                if True:
                    pass



                    if False:
                        begin_wf_item (index = 112, name = "массовые запасы нефти")
                        grid_property_export_gridecl_format (grid=find_object (name="DynamicModel (Гидродинамическая модель)",
                              type="gt_tnav_grid_3d_data"),
                              grid_property=find_object (name="OIPM",
                              type="gt_tnav_resource_cube_3d_data"),
                              file_name="Example/02_oil/property_directory/OIPM.INC",
                              use_precision=False,
                              precision=0,
                              keyword="OIPM",
                              inactive_placeholder="0",
                              separate_by_comment=False,
                              units_system="metric")
                        end_wf_item (index = 112)


                    if False:
                        begin_wf_item (index = 113, name = "проницаемость")
                        grid_property_export_gridecl_format (grid=find_object (name="DynamicModel (Гидродинамическая модель)",
                              type="gt_tnav_grid_3d_data"),
                              grid_property=find_object (name="INIT_PERMX",
                              type="gt_tnav_cube_3d_data"),
                              file_name="Example/02_oil/property_directory/PERMX.INC",
                              use_precision=False,
                              precision=0,
                              keyword="INIT_PERMX",
                              inactive_placeholder="0",
                              separate_by_comment=False,
                              units_system="metric")
                        end_wf_item (index = 113)


                    if False:
                        begin_wf_item (index = 114, name = "регионы")
                        grid_property_export_gridecl_format (grid=find_object (name="DynamicModel (Гидродинамическая модель)",
                              type="gt_tnav_grid_3d_data"),
                              grid_property=find_object (name="EQLNUM",
                              type="gt_tnav_cube_3d_data"),
                              file_name="Example/02_oil/regions_directory/EQLNUM.INC",
                              use_precision=False,
                              precision=0,
                              keyword="EQLNUM",
                              inactive_placeholder="0",
                              separate_by_comment=False,
                              units_system="metric")
                        end_wf_item (index = 114)



                end_wf_item (index = 111)


            if False:
                begin_wf_item (index = 116, name = "Газ")
                workflow_folder ()
                if True:
                    pass



                    if False:
                        begin_wf_item (index = 117, name = "массовые запасы газа")
                        grid_property_export_gridecl_format (grid=find_object (name="DynamicModel (Гидродинамическая модель)",
                              type="gt_tnav_grid_3d_data"),
                              grid_property=find_object (name="GIPM",
                              type="gt_tnav_resource_cube_3d_data"),
                              file_name="Example/01_gas/property_directory/GIPM.INC",
                              use_precision=False,
                              precision=0,
                              keyword="GIPM",
                              inactive_placeholder="0",
                              separate_by_comment=False,
                              units_system="metric")
                        end_wf_item (index = 117)


                    if False:
                        begin_wf_item (index = 118, name = "проницаемость")
                        grid_property_export_gridecl_format (grid=find_object (name="DynamicModel (Гидродинамическая модель)",
                              type="gt_tnav_grid_3d_data"),
                              grid_property=find_object (name="INIT_PERMX",
                              type="gt_tnav_cube_3d_data"),
                              file_name="Example/01_gas/property_directory/PERMX.INC",
                              use_precision=False,
                              precision=0,
                              keyword="INIT_PERMX",
                              inactive_placeholder="0",
                              separate_by_comment=False,
                              units_system="metric")
                        end_wf_item (index = 118)


                    if False:
                        begin_wf_item (index = 119, name = "регионы")
                        grid_property_export_gridecl_format (grid=find_object (name="DynamicModel (Гидродинамическая модель)",
                              type="gt_tnav_grid_3d_data"),
                              grid_property=find_object (name="EQLNUM",
                              type="gt_tnav_cube_3d_data"),
                              file_name="Example/01_gas/regions_directory/EQLNUM.INC",
                              use_precision=False,
                              precision=0,
                              keyword="EQLNUM",
                              inactive_placeholder="0",
                              separate_by_comment=False,
                              units_system="metric")
                        end_wf_item (index = 119)



                end_wf_item (index = 116)



        end_wf_item (index = 101)


    if False:
        begin_wf_item (index = 122, name = "Casekill", collapsed = True)
        workflow_folder ()
        if True:
            pass



            if False:
                begin_wf_item (index = 123, is_custom_code = True, name = "Manager param")

                class Manager:
                    def __init__(self, variable_file: str, config_varibale: str, default_variables: str):
                        self.variable_file = Manager.check_file(variable_file)
                        self.config_variable_file = Manager.check_file(config_varibale)
                        self.default_variables_file = Manager.check_file(default_variables)
                        self.load_variable(), self.load_config_variable()

                    @staticmethod
                    def check_file(file_path: str):
                        """
                        Проверяет, что файл существует, является JSON файлом и другие условия.
                        """
                        if not os.path.exists(file_path):
                            raise FileNotFoundError(f"Файл {file_path} не найден.")
                        if not file_path.lower().endswith('.json'):
                            raise ValueError(f"Файл {file_path} не является JSON файлом.")
                        if not os.access(file_path, os.R_OK):
                            raise PermissionError(f"Файл {file_path} не доступен для чтения.")
                        if os.path.getsize(file_path) == 0:
                            raise ValueError(f"Файл {file_path} пустой.")
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                json.load(f)
                        except json.JSONDecodeError:
                            raise ValueError(f"Файл {file_path} содержит некорректный JSON.")
                        return file_path

                    @staticmethod
                    def convert_value_from_str(value, variable_type):
                        """
                        Преобразует значение в соответствующий тип данных.
                        :param value: Значение переменной.
                        :param variable_type: Тип данных переменной.
                        :return: Преобразованное значение.
                        """
                        if value == '':
                            return None
                        if variable_type == "int":
                            return int(value)
                        elif variable_type == "float":
                            return float(value)
                        elif variable_type == "list":
                            return json.loads(value)
                        elif variable_type == "datetime":
                            # Если Python имеет версию ранее 3.7
                            return datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                            # Если Python имеет версию позже 3.7
                            # return datetime.fromisoformat(value)
                        elif variable_type == 'bool':
                            return bool(value)
                        else:
                            return value

                    @staticmethod
                    def convert_value_to_str(value, variable_type):
                        """
                        Преобразует значение в строку для сохранения в JSON.
                        :param value: Значение переменной.
                        :param variable_type: Тип данных переменной.
                        :return: Преобразованное значение в строку.
                        """
                        if variable_type == "list":
                            return json.dumps(value)
                        elif variable_type == "datetime":
                            # Если Python имеет версию ранее 3.7
                            return value.strftime('%Y-%m-%d %H:%M:%S')
                            # Если Python имеет версию позже 3.7
                            # return value.isoformat()
                        else:
                            return str(value)

                    def load_variable(self):
                        """
                        Чтение json файла со значениями переменных
                        :return: None
                        """
                        with open(self.variable_file, 'r', encoding='utf-8') as file:
                            variable = json.load(file)
                        self.variable = variable

                    def load_config_variable(self):
                        """
                        Чтение json файла с настройками переменными
                        :return: None
                        """
                        with open(self.config_variable_file, 'r', encoding='utf-8') as file:
                            config_variable = json.load(file)
                        self.config_variable = config_variable

                    @property
                    def info_variable(self):
                        """
                        :return: Dict со значениями переменных
                        """
                        return self.variable

                    @property
                    def info_variable_settings(self):
                        """
                        :return: Dict настроек переменных
                        """
                        return self.config_variable

                    def get_variable(self, key: str):
                        """
                        Получает значение переменной по указанным ключам.
                        :param key: Ключ для доступа к переменной.
                        :return: Значение переменной.
                        """
                        if key in self.variable:
                            if isinstance(key, str):
                                return Manager.convert_value_from_str(self.variable[key], self.config_variable[key]['type'])
                            else:
                                raise AssertionError("ключ {key} к переменной должен задаваться через ''")
                        else:
                            raise AssertionError(f"Переменная '{key}' не существует")

                    def set_variable(self, key: str, value):
                        """
                        Устанавливает значение переменной по указанным ключам.
                        :param key: Ключи для доступа к переменной.
                        :param value: Новое значение переменной.
                        """
                        if key in self.variable:
                            if isinstance(key, str):
                                self.variable[key] = Manager.convert_value_to_str(value, self.config_variable[key]['type'])
                                self.save_variable()
                            else:
                                raise AssertionError("ключ {key} к переменной должен задаваться через ''")
                        else:
                            raise AssertionError(f"Переменная '{key}' не существует")

                    def save_variable(self):
                        """
                        Сохраняет изменения в JSON файл.
                        """
                        with open(self.variable_file, 'w', encoding='utf-8') as file:
                            json.dump(self.variable, file, ensure_ascii=False, indent=4)

                    def help_variable(self, variable_name):
                        """
                        Получает информацию о переменной по её имени.
                        :param variable_name: Имя переменной.
                        :return: Информация о переменной.
                        """
                        variable_info = self.config_variable.get(variable_name, "Переменная не найдена")
                        if isinstance(variable_info, dict):
                            print(f"Информация о переменной '{variable_name}':")
                            for key, value in variable_info.items():
                                print(f" {key}: {value}")
                        else:
                            print(variable_info)
                        return variable_info

                    def help_field(self, field_name):
                        """
                        Получает информацию о всех переменных по указанному полю.
                        :param field_name: Имя поля (например, 'description').
                        :return: Словарь с информацией о переменных.
                        """
                        field_info = {}
                        for variable_name, variable_info in self.config_variable.items():
                            if field_name in variable_info:
                                field_info[variable_name] = variable_info[field_name]
                                print(f"{variable_name}: {variable_info[field_name]}")
                        return field_info

                    def reset_to_default(self):
                        """
                        Сбрасывает все изменения и возвращает переменные к значениям по умолчанию.
                        """
                        shutil.copyfile(self.default_variables_file, self.variable_file)
                        self.load_variable()
                end_wf_item (index = 123)


            if False:
                begin_wf_item (index = 124, is_custom_code = True, name = "start")
                def run_script(script_path):
                    runpy.run_path(script_path, run_name='__main__')

                if __name__ == "__main__":
                    script_path = "D:\14. Кейсы\005. IRM\app\smartcase.py"  # Укажите путь к файлу script_to_run.py
                    run_script(script_path)
                end_wf_item (index = 124)


            if False:
                begin_wf_item (index = 125, name = "Опция \"Нефть\"")
                workflow_folder ()
                if True:
                    pass



                    if False:
                        begin_wf_item (index = 126, is_custom_code = True, name = "Определение переменных моделей")
                        manager_oil = Manager(
                        	variable_file = r"D:\Orher\oil\variable.json", 
                        	config_varibale = r"D:\14. Кейсы\005. IRM\models\Example\02_oil\variable_directory\conditions_variable.json", 
                        	default_variables = r"D:\14. Кейсы\005. IRM\models\Example\02_oil\variable_directory\by_default_variables.json"
                        	)

                        # Вывод дополнительной информации о характеристик переменных
                        print(manager_oil.help_field('description'))
                        #		Доступно 
                        #		"type", Тип данных
                        #		"measure", Едицина измерения
                        #		"description", Описание переменной
                        #		"status_empty", Статус переменной, если переменная моежт быть пустой - True,иначе False
                        #		"status_variable", Статус переменной, для чего используется переменная
                        #		"limit" Верхние и нижние границы диапазаонов вариации переменных

                        '''

                        			Описание принимаемых атрибутов класса Manager

                        variable_file = Заполняемый шаблон значений переменных. Который будет использован в качестве переменной для запуска main.exe
                        config_varibale = Json файл с описанием переменных, типы, ограничения (менять не рекомендуется, т.к. main.exe ссылается своему config_varibale)
                        default_variables = Пустой шаблон для сброса изменений значений переменных установленных в variable_file

                        '''


                        def get(key:str) -> None:
                        	'''
                        	Вывод сообщения 
                        	'''
                        	print(f"Установленное значение {key} = {manager_oil.get_variable(key)}")


                        '''

                        			Обязательные параметры расчета

                        '''

                        # Устанавливаем название расчета
                        manager_oil.set_variable('name', 'OIL')
                        get('name')

                        # Устанавливаем начало расчета (согласно начальному времени расчета в проекте Дизайнера моеделей)
                        manager_oil.set_variable('start', datetime(year=2024, month=1, day=1))
                        get('start')

                        # Устанавливаем время расчета (согласно времени расчета в проекте Дизайнера моеделей, в годах)
                        manager_oil.set_variable('duration', 20)
                        get('duration')

                        # Устанавливаем абсолютный путь к выгруженным кубам свойств. В данном случае к директории с PERMX, OIPM
                        manager_oil.set_variable('property_directory', r"D:\14. Кейсы\005. IRM\models\Example\02_oil\property_directory")
                        get('property_directory')

                        # Устанавливаем абсолютный путь к выгрузке координатной сетки
                        manager_oil.set_variable('coord_directory', r"D:\14. Кейсы\005. IRM\models\Example\02_oil\coord_directory")
                        get('coord_directory')

                        # Устанавливаем абсолютный путь для сохранения результатов расчета скрипта
                        manager_oil.set_variable('project_directory', r"D:\14. Кейсы\005. IRM\models\Example\02_oil\project_directory")
                        get('project_directory')

                        '''

                        			Необязательные параметры

                        '''

                        # Устанавливаем абсолютный путь к выгруженныму кубу регионов. В данном случае к директории с EQLNUM
                        manager_oil.set_variable('regions_directory', r"D:\14. Кейсы\005. IRM\models\Example\02_oil\regions_directory")
                        get('regions_directory')

                        # Устанавливаем абсолютный путь к выгруженныму кубу фильтра. В данном случае к директории, которая пустая (опция расчета с фильтром не будет использована)
                        manager_oil.set_variable('mask_directory', r"D:\14. Кейсы\005. IRM\models\Example\02_oil\mask_directory")
                        get('mask_directory')

                        # Устанавливаем путь к растровой карте поверхности.
                        # ВАЖНО!: Путь не должен содержать русскоязычные символы (из-за особенностей работы cv2)
                        manager_oil.set_variable('map_directory', r"D:\maps")
                        get('map_directory')

                        # Дневная задержка по мобилизации БУ от начала времени расчета
                        manager_oil.set_variable('relative_mob_start', 0)
                        get('relative_mob_start')

                        # Дневная задержка по запуску кейса от начала времени расчета
                        # Допускаем, что от начала расчета и с момента готовности нефтяной инфраструктуры должно пройти 1.5 лет
                        time_pause = int(1.5*365)
                        manager_oil.set_variable('relative_pmr_start', time_pause)
                        get('relative_pmr_start')


                        '''

                        			Параметры оптимизации 

                        '''

                        # Устанавливаем параметр фильтрации для нормализованного куба PERMX x OIPM (Считается внутри main.exe)
                        manager_oil.set_variable('selection', int(SELECTION_OIL))
                        get('selection')

                        # Устанавливаем ограничение на максимальное количество КП 
                        manager_oil.set_variable('wp_max', int(WP_MAX_OIL))
                        get('wp_max')

                        # Устанавливаем ограничение на максимальное количество стволов 
                        # wt_max - 1: заканчивание ГС, wt_max - 2: заканчивание двухстволка, wt_max = 3.заканчивание МЗС
                        manager_oil.set_variable('wt_max', int(WT_MAX_OIL))
                        get('wt_max')

                        # Устанавливаем количество БУ под мобилиацию
                        manager_oil.set_variable('mobil_dr', int(MOBIL_BU_OIL))
                        get('mobil_dr')

                        # Вывод всех переменных
                        print(manager_oil.info_variable)

                        # Вывод всех характеристик переменных
                        print(manager_oil.info_variable_settings)
                        end_wf_item (index = 126)


                    begin_wf_item (index = 127, name = "Пояснения ")
                    comment_text ("""
Путь к variable_file должен:
1) Совпадать с self.variable_file класса Manager для объекта manager_oil
2) Не должен содержать русскоязычные символы (из-за особеностей кодировщика)
""")
                    end_wf_item (index = 127)


                    if False:
                        begin_wf_item (index = 128, name = "Запуск main.exe")
                        execute_external_program (program="D:/14. Кейсы/005. IRM/res/SmartCase/SmartCase.exe",
                              arguments="\"D:\\Orher\\oil\\variable.json\"",
                              use_working_directory=False,
                              working_directory="")
                        end_wf_item (index = 128)


                    if False:
                        begin_wf_item (index = 129, name = "Работа с скважинами")
                        workflow_folder ()
                        if True:
                            pass



                            if False:
                                begin_wf_item (index = 130, is_custom_code = True, name = "Удалить")
                                print(get_all_wells())
                                for w in get_all_wells():
                                	delete_well (name=w.name)
                                end_wf_item (index = 130)


                            if False:
                                begin_wf_item (index = 131)
                                wells_import_welltrack_format (wells=find_object (name="Wells",
                                      type="gt_wells_entity"),
                                      trajectories=find_object (name="Trajectories",
                                      type="Trajectories"),
                                      do_remove_existing_wells=False,
                                      well_searcher="name",
                                      splitter=True,
                                      file_names=["Example/02_oil/project_directory/current/welltrac.INC"],
                                      splitter2=True,
                                      use_oem_encoding=False,
                                      add_zero_point=False,
                                      invert_z=False,
                                      use_well_filter=False,
                                      result_well_filter=find_object (name="Well Filter 1",
                                      type="WellFilter"),
                                      file_datum_info=CrsInfo (crs_type="not_specified",
                                      crs_code=None,
                                      crs_name="",
                                      crs_proj_string=None,
                                      datum_name=None,
                                      datum_bounds_inited=False,
                                      datum_bounds_min_x=0,
                                      datum_bounds_max_x=0,
                                      datum_bounds_min_y=0,
                                      datum_bounds_max_y=0,
                                      datum_is_in_proj4=False),
                                      xy_units_system="si",
                                      z_units_system="si",
                                      use_xy_units=True,
                                      xy_units="metres",
                                      use_z_units=True,
                                      z_units="metres")
                                end_wf_item (index = 131)


                            if False:
                                begin_wf_item (index = 132, is_custom_code = True, name = "Перфорации")
                                import pandas as pd
                                from datetime import datetime

                                with open(r"D:\14. Кейсы\005. IRM\models\Example\02_oil\project_directory\current\well_events.INC", 'r', encoding='utf-8') as file:
                                		lines = file.readlines()
                                data = []
                                for line in lines[1:]:
                                	parts = line.strip().split('\t')
                                	if len(parts) == 6:
                                		well, date_str, event, top_depth, bottom_depth, depth_type = parts
                                		day, month, year = map(int, date_str.split('.'))
                                		if ':' in well:
                                			well_name, well_number = well.split(':')
                                			well_number = int(well_number)
                                		else:
                                			well_name = well
                                			well_number = 0
                                	data.append([well_name, well_number, event, top_depth, bottom_depth, depth_type, day, month, year])
                                df = pd.DataFrame(data, columns=['Скважина', 'Номер ствола', 'Событие', 'Кровля', 'Подошва', 'Глубина', 'День', 'Месяц', 'Год'])

                                print(df)

                                T = get_all_wells_structure_tables ()[0]
                                for well_name in df['Скважина'].unique():
                                	well_records = df[df['Скважина'] == well_name]
                                	for index, row in well_records.iterrows():
                                		print(f"perforation well {well_name}:{row['Номер ствола']} c {row['Кровля']} м. до {row['Подошва']} м. по MD")
                                		date_ = datetime(row['Год'], row['Месяц'], row['День'])
                                		w=get_well_by_name(name=well_name)
                                		new_record = T.add_record (well=w, branch_num=row['Номер ствола'], date=date_)
                                		new_record.set_value (type='event', value='Perforation')
                                		new_record.set_value (type='top_depth', value=float(row['Кровля']))
                                		new_record.set_value (type='bottom_depth', value=float(row['Подошва']))
                                		new_record.set_value (type='skin', value=0)
                                		new_record.set_value (type='multiplier', value=1)
                                		new_record.set_value (type='depth_type', value='MD')

                                end_wf_item (index = 132)



                        end_wf_item (index = 129)



                end_wf_item (index = 125)


            if False:
                begin_wf_item (index = 135, name = "Опция \"Газ\"")
                workflow_folder ()
                if True:
                    pass



                    if False:
                        begin_wf_item (index = 136, is_custom_code = True, name = "Определение переменных моделей")
                        manager_gas = Manager(
                        	variable_file = r"D:\Orher\gas\variable.json", 
                        	config_varibale = r"D:\14. Кейсы\005. IRM\models\Example\01_gas\variable_directory\conditions_variable.json", 
                        	default_variables = r"D:\14. Кейсы\005. IRM\models\Example\01_gas\variable_directory\by_default_variables.json"
                        	)

                        # Вывод дополнительной информации о характеристик переменных
                        print(manager_gas.help_field('description'))
                        #		Доступно 
                        #		"type", Тип данных
                        #		"measure", Едицина измерения
                        #		"description", Описание переменной
                        #		"status_empty", Статус переменной, если переменная моежт быть пустой - True,иначе False
                        #		"status_variable", Статус переменной, для чего используется переменная
                        #		"limit" Верхние и нижние границы диапазаонов вариации переменных

                        '''

                        			Описание принимаемых атрибутов класса Manager

                        variable_file = Заполняемый шаблон значений переменных. Который будет использован в качестве переменной для запуска main.exe
                        config_varibale = Json файл с описанием переменных, типы, ограничения (менять не рекомендуется, т.к. main.exe ссылается своему config_varibale)
                        default_variables = Пустой шаблон для сброса изменений значений переменных установленных в variable_file

                        '''


                        def get(key:str) -> None:
                        	'''
                        	Вывод сообщения 
                        	'''
                        	print(f"Установленное значение {key} = {manager_gas.get_variable(key)}")


                        '''

                        			Обязательные параметры расчета

                        '''

                        # Устанавливаем название расчета
                        manager_gas.set_variable('name', 'GAS')
                        get('name')

                        # Устанавливаем начало расчета (согласно начальному времени расчета в проекте Дизайнера моеделей)
                        manager_gas.set_variable('start', datetime(year=2024, month=1, day=1))
                        get('start')

                        # Устанавливаем время расчета (согласно времени расчета в проекте Дизайнера моеделей, в годах)
                        manager_gas.set_variable('duration', 20)
                        get('duration')

                        # Устанавливаем абсолютный путь к выгруженным кубам свойств. В данном случае к директории с PERMX, GIPM
                        manager_gas.set_variable('property_directory', r"D:\14. Кейсы\005. IRM\models\Example\01_gas\property_directory")
                        get('property_directory')

                        # Устанавливаем абсолютный путь к выгрузке координатной сетки
                        manager_gas.set_variable('coord_directory', r"D:\14. Кейсы\005. IRM\models\Example\01_gas\coord_directory")
                        get('coord_directory')

                        # Устанавливаем абсолютный путь для сохранения результатов расчета скрипта
                        manager_gas.set_variable('project_directory', r"D:\14. Кейсы\005. IRM\models\Example\01_gas\project_directory")
                        get('project_directory')

                        '''

                        			Необязательные параметры

                        '''

                        # Устанавливаем абсолютный путь к выгруженныму кубу регионов. В данном случае к директории с EQLNUM
                        manager_gas.set_variable('regions_directory', r"D:\14. Кейсы\005. IRM\models\Example\01_gas\regions_directory")
                        get('regions_directory')

                        # Устанавливаем абсолютный путь к выгруженныму кубу фильтра. В данном случае к директории, которая пустая (опция расчета с фильтром не будет использована)
                        manager_gas.set_variable('mask_directory', r"D:\14. Кейсы\005. IRM\models\Example\01_gas\mask_directory")
                        get('mask_directory')

                        # Устанавливаем путь к растровой карте поверхности.
                        # ВАЖНО!: Путь не должен содержать русскоязычные символы (из-за особенностей работы cv2)
                        manager_gas.set_variable('map_directory', r"D:\maps")
                        get('map_directory')

                        # Дневная задержка по мобилизации БУ от начала времени расчета
                        manager_gas.set_variable('relative_mob_start', 0)
                        get('relative_mob_start')

                        # Дневная задержка по запуску кейса от начала времени расчета
                        # Допускаем, что от начала расчета и с момента готовности газовой инфраструктуры должно пройти 2 года
                        time_pause = int(2*365)
                        manager_gas.set_variable('relative_pmr_start', time_pause)
                        get('relative_pmr_start')

                        '''

                        			Параметры оптимизации 

                        '''

                        # Устанавливаем параметр фильтрации для нормализованного куба PERMX x GIPM (Считается внутри main.exe)
                        manager_gas.set_variable('selection', int(SELECTION_GAS))
                        get('selection')

                        # Устанавливаем ограничение на максимальное количество КП 
                        manager_gas.set_variable('wp_max', int(WP_MAX_GAS))
                        get('wp_max')

                        # Устанавливаем ограничение на максимальное количество стволов 
                        # wt_max - 1: заканчивание ГС, wt_max - 2: заканчивание двухстволка, wt_max = 3.заканчивание МЗС
                        manager_gas.set_variable('wt_max', int(WT_MAX_GAS))
                        get('wt_max')

                        # Устанавливаем количество БУ под мобилиацию
                        manager_gas.set_variable('mobil_dr', int(MOBIL_BU_GAS))
                        get('mobil_dr')

                        # Вывод всех переменных
                        print(manager_gas.info_variable)

                        # Вывод всех характеристик переменных
                        print(manager_gas.info_variable_settings)
                        end_wf_item (index = 136)


                    begin_wf_item (index = 137, name = "Пояснения ")
                    comment_text ("""
Путь к variable_file должен:
1) Совпадать с self.variable_file класса Manager для объекта manager_gas
2) Не должен содержать русскоязычные символы (из-за особеностей кодировщика)
""")
                    end_wf_item (index = 137)


                    if False:
                        begin_wf_item (index = 138, name = "Запуск main.exe")
                        execute_external_program (program="D:/14. Кейсы/005. IRM/res/SmartCase/SmartCase.exe",
                              arguments="\"D:\\Orher\\gas\\variable.json\"",
                              use_working_directory=False,
                              working_directory="")
                        end_wf_item (index = 138)


                    if False:
                        begin_wf_item (index = 139, name = "Работа с скважинами")
                        workflow_folder ()
                        if True:
                            pass



                            begin_wf_item (index = 140)
                            comment_text ("""
Необходимо снять галочку над командой \"Удалить\", в случае запуска расчета нефтяного и газового кейса одновременно
""")
                            end_wf_item (index = 140)


                            if False:
                                begin_wf_item (index = 141, is_custom_code = True, name = "Удалить")
                                print(get_all_wells())
                                for w in get_all_wells():
                                	delete_well (name=w.name)
                                end_wf_item (index = 141)


                            if False:
                                begin_wf_item (index = 142)
                                wells_import_welltrack_format (wells=find_object (name="Wells",
                                      type="gt_wells_entity"),
                                      trajectories=find_object (name="Trajectories",
                                      type="Trajectories"),
                                      do_remove_existing_wells=False,
                                      well_searcher="name",
                                      splitter=True,
                                      file_names=["Example/01_gas/project_directory/current/welltrac.INC"],
                                      splitter2=True,
                                      use_oem_encoding=False,
                                      add_zero_point=False,
                                      invert_z=False,
                                      use_well_filter=False,
                                      result_well_filter=find_object (name="Well Filter 1",
                                      type="WellFilter"),
                                      file_datum_info=CrsInfo (crs_type="not_specified",
                                      crs_code=None,
                                      crs_name="",
                                      crs_proj_string=None,
                                      datum_name=None,
                                      datum_bounds_inited=False,
                                      datum_bounds_min_x=0,
                                      datum_bounds_max_x=0,
                                      datum_bounds_min_y=0,
                                      datum_bounds_max_y=0,
                                      datum_is_in_proj4=False),
                                      xy_units_system="si",
                                      z_units_system="si",
                                      use_xy_units=True,
                                      xy_units="metres",
                                      use_z_units=True,
                                      z_units="metres")
                                end_wf_item (index = 142)


                            if False:
                                begin_wf_item (index = 143, is_custom_code = True, name = "Перфорации")
                                import pandas as pd
                                from datetime import datetime

                                with open(r"D:\14. Кейсы\005. IRM\models\Example\01_gas\project_directory\current\well_events.INC", 'r', encoding='utf-8') as file:
                                		lines = file.readlines()
                                data = []
                                for line in lines[1:]:
                                	parts = line.strip().split('\t')
                                	if len(parts) == 6:
                                		well, date_str, event, top_depth, bottom_depth, depth_type = parts
                                		day, month, year = map(int, date_str.split('.'))
                                		if ':' in well:
                                			well_name, well_number = well.split(':')
                                			well_number = int(well_number)
                                		else:
                                			well_name = well
                                			well_number = 0
                                	data.append([well_name, well_number, event, top_depth, bottom_depth, depth_type, day, month, year])
                                df = pd.DataFrame(data, columns=['Скважина', 'Номер ствола', 'Событие', 'Кровля', 'Подошва', 'Глубина', 'День', 'Месяц', 'Год'])

                                print(df)

                                T = get_all_wells_structure_tables ()[0]
                                for well_name in df['Скважина'].unique():
                                	well_records = df[df['Скважина'] == well_name]
                                	for index, row in well_records.iterrows():
                                		print(f"perforation well {well_name}:{row['Номер ствола']} c {row['Кровля']} м. до {row['Подошва']} м. по MD")
                                		date_ = datetime(row['Год'], row['Месяц'], row['День'])
                                		w=get_well_by_name(name=well_name)
                                		new_record = T.add_record (well=w, branch_num=row['Номер ствола'], date=date_)
                                		new_record.set_value (type='event', value='Perforation')
                                		new_record.set_value (type='top_depth', value=float(row['Кровля']))
                                		new_record.set_value (type='bottom_depth', value=float(row['Подошва']))
                                		new_record.set_value (type='skin', value=0)
                                		new_record.set_value (type='multiplier', value=1)
                                		new_record.set_value (type='depth_type', value='MD')

                                end_wf_item (index = 143)



                        end_wf_item (index = 139)



                end_wf_item (index = 135)


            if False:
                begin_wf_item (index = 146)
                schedule_rule_well_structure (schedule_strategy=find_object (name="Case",
                      type="gt_schedule_rules_data"),
                      use_rule_name=True,
                      rule_name="Ввод конструкции скважин",
                      object_set_type="all_objects",
                      object_set_name="Все скважины",
                      params_table=[{"table_name" : find_object (name="Конструкция скважин",
                      type="gt_wells_events_data")}])
                end_wf_item (index = 146)


            if False:
                begin_wf_item (index = 147)
                schedule_import (reload_all=True,
                      file_name="Example/02_oil/project_directory/current/schedule.sch",
                      strategy=find_object (name="Case",
                      type="gt_schedule_rules_data"),
                      starting_date=datetime (year=2024,
                      month=1,
                      day=1,
                      hour=0,
                      minute=0,
                      second=0),
                      select_import_mode="Convert Model",
                      result_well_events=find_object (name="Конструкция скважин",
                      type="gt_wells_events_data"),
                      additional_options=True,
                      apply_welspecs_later=False)
                end_wf_item (index = 147)


            if False:
                begin_wf_item (index = 148)
                schedule_import (reload_all=False,
                      file_name="Example/01_gas/project_directory/current/schedule.sch",
                      strategy=find_object (name="Case",
                      type="gt_schedule_rules_data"),
                      starting_date=datetime (year=2024,
                      month=1,
                      day=1,
                      hour=0,
                      minute=0,
                      second=0),
                      select_import_mode="Convert Model",
                      result_well_events=find_object (name="Конструкция скважин",
                      type="gt_wells_events_data"),
                      additional_options=True,
                      apply_welspecs_later=False)
                end_wf_item (index = 148)


            if False:
                begin_wf_item (index = 149, name = "Управление скважинами")
                schedule_rule_add_apply_script (schedule_strategy=find_object (name="Case",
                      type="gt_schedule_rules_data"),
                      date_time=datetime (year=2024,
                      month=1,
                      day=1,
                      hour=0,
                      minute=0,
                      second=0),
                      use_rule_name=True,
                      rule_name="Управление нефтяными скважинами",
                      file_name="control",
                      function_name="control_wells",
                      variables_table=[{"variable" : "depression_oil", "value" : resolve_variables_in_string (string_with_variables="@DEP_OIL@",
                      variables=variables)}, {"variable" : "limit_oil_liq_well", "value" : "500"}, {"variable" : "depression_gas", "value" : resolve_variables_in_string (string_with_variables="@DEP_GAS@",
                      variables=variables)}, {"variable" : "limit_gas_gas_well", "value" : "500000"}],
                      script_text="	for group in get_all_groups ( ):\n		if group.name == \'OIL\':\n			for w in group.all_wells:\n				# Получаем все скважины на нефть\n				if w.is_stopped ( ):\n					continue\n				BHP_ = wbp9[w] - depression_oil\n				print(f\'Настройка скважины {w.name} Зайбоное давление - {BHP_}\')\n				if BHP_ < 40:\n					print(f\'Отключаем скважину {w.name} Зайбоное давление - {BHP_} ниже 40 бар\')\n					set_well_status (w.name, status = \'stop\')\n					continue\n				\n				add_keyword(\n					\"\"\"\n					WCONPROD\n					\"\"\"+w.name+\"\"\" OPEN LRAT 3* \"\"\"+ str(limit_oil_liq_well) +\"\"\" * \"\"\"+ str(BHP_) +\"\"\" /\n					/\n					\"\"\"\n					)\n				print(f\'Задаем ограничения на скважину {w.name}\')\n				add_keyword(\n					\"\"\"\n					WECON\n					\"\"\"+w.name+\"\"\" 10 * 0.98 8000 * CON+ /\n					/\n					\"\"\"\n					)\n		elif group.name == \'GAS\':\n			for w in group.all_wells:\n				# Получаем все скважины на нефть\n				if w.is_stopped ( ):\n					continue\n				BHP_ = wbp9[w] - depression_gas\n				print(f\'Настройка скважины {w.name} Зайбоное давление - {BHP_}\')\n				if BHP_ < 40:\n					print(f\'Отключаем скважину {w.name} Зайбоное давление - {BHP_} ниже 40 бар\')\n					set_well_status (w.name, status = \'stop\')\n					continue\n				\n				add_keyword(\n					\"\"\"\n					WCONPROD\n					\"\"\"+w.name+\"\"\" OPEN GRAT 2* \"\"\"+ str(limit_gas_gas_well) +\"\"\" 2* \"\"\"+ str(BHP_) +\"\"\" /\n					/\n					\"\"\"\n					)\n				print(f\'Задаем ограничения на скважину {w.name}\')\n				add_keyword(\n					\"\"\"\n					WECON\n					\"\"\"+w.name+\"\"\" * 10000 2* 0.005 CON+ /\n					/\n					\"\"\"\n					)\n		\n")
                end_wf_item (index = 149)



        end_wf_item (index = 122)


