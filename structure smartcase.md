classDiagram
direction BT
class node7 {
   __init__(self, variable_file) 
   create_project_structure(self) 
   move_current_to_history(self) 
   is_current_folder_not_empty(self) 
   clear_current_folder(self) 
   create_json_file(self) 
   __call__(self, *args, **kwargs) 
}
class node2 {
   __init__(self, variable_file: str) 
   resource_path(relative_path) 
   check_file(file_path: str) 
   convert_value_from_str(value, variable_type) 
   convert_value_to_str(value, variable_type) 
   load_variable(self) 
   load_config_variable(self) 
   info_variable(self) 
   info_variable_settings(self) 
   get_variable(self, key: str) 
   set_variable(self, key: str, value) 
   save_variable(self) 
   help_variable(self, variable_name) 
   help_field(self, field_name) 
   reset_to_default(self) 
}
class node6 {
   __init__(self, variable_file) 
   define_attributes(self) 
   validate_variable(self, key, value) 
   validate_between_variables(self) 
   __call__(self) 
}
class node3 {
   __init__(self, variable_file) 
   __read_files(self) 
   __output_files(self) 
   __call__(self, *args, **kwargs) 
   get_coords(self) 
   get_values(self) 
   get_regions(self) 
   get_current_value(self, coord: np.ndarray) 
   get_current_values(self, coords: np.ndarray) 
}
class node9 {
   __init__(self, variable_file) 
   __normalize(self) 
   __selection(self) 
   __density(self) 
   __call__(self, *args, **kwargs) 
   get_dfn_coords(self) 
   get_dfn_values(self) 
   get_df_regions(self) 
   get_ddnf_coords(self) 
}
class node0 {
   __init__(self, variable_file) 
   __create_date(self) 
   __save_cover(self) 
   __create_schedule(self) 
   get_dict_date_drill(self) 
   get_dict_date_input(self) 
   get_fwp_names(self) 
   get_fwp_coords(self) 
   get_dict_fwp_coords(self) 
   get_fwp_values(self) 
   get_dict_fwp_values(self) 
   __call__(self, *args, **kwargs) 
}
class node10 {
   __init__(self, variable_file) 
   __create_map(self) 
   __create_w_pad(self) 
   __wp_visual(self) 
   __call__(self, *args, **kwargs) 
   get_wp_names(self) 
   get_n_wp_points(self) 
   get_a_wp_points(self) 
   get_wp_values(self) 
   get_dict_n_wp_points(self) 
   get_dict_a_wp_points(self) 
   get_dict_wp_values(self) 
   get_n_wp_wz_points(self) 
   get_a_wp_wz_points(self) 
   get_wp_wz_values(self) 
   get_wp_wz_names(self) 
   get_dict_n_wp_wz_points(self) 
   get_dict_a_wp_wz_points(self) 
   get_dict_wp_wz_values(self) 
   get_dict_wp_wz_names(self) 
   get_wp_area(self) 
}
class node8 {
   __init__(self, variable_file) 
   __create_wells(self) 
   __recreate_wells(self) 
   __wt_visual(self) 
   __call__(self, *args, **kwargs) 
   get_dict_wt_trajectories(self) 
   get_dict_wt_lenghts_vs(self) 
   get_dict_wt_lenghts_gs(self) 
   get_dict_wt_lenghts_ds(self) 
   get_dict_wt_quantity_ds(self) 
   get_dict_wt_wp_names(self) 
   get_dict_wp_wt_names(self) 
   get_dict_wt_values(self) 
   get_dict_wt_wz_points(self) 
   get_dict_wt_wz_weight(self) 
   get_dict_wt_ds_names(self) 
   get_dict_wtds_md_start(self) 
   get_dict_wtds_md_end(self) 
}
class node5 {
   __init__(self, variable_file) 
   __definition_area(self) 
   __create_w_zone(self) 
   __create_w_zone_regions(self) 
   __wz_visual(self) 
   __call__(self, *args, **kwargs) 
   get_area(self) 
   get_w_area(self) 
   get_wz_names(self) 
   get_n_wz_points(self) 
   get_a_wz_points(self) 
   get_wz_values(self) 
   get_dict_n_wz_points(self) 
   get_dict_a_wz_points(self) 
   get_dict_wz_values(self) 
   get_n_belong_points(self) 
   get_a_belong_points(self) 
   get_belong_values(self) 
   get_dict_n_belong_points(self) 
   get_dict_a_belong_points(self) 
   get_dict_belong_values(self) 
}
class object {
   __class__(self: _T) 
   __class__(self, __type: Type[object]) 
   __init__(self) 
   __new__(cls: Type[_T]) 
   __setattr__(self, name: str, value: Any) 
   __eq__(self, o: object) 
   __ne__(self, o: object) 
   __str__(self) 
   __repr__(self) 
   __hash__(self) 
   __format__(self, format_spec: str) 
   __getattribute__(self, name: str) 
   __delattr__(self, name: str) 
   __sizeof__(self) 
   __reduce__(self) 
   __reduce_ex__(self, protocol: SupportsIndex) 
   __reduce_ex__(self, protocol: int) 
   __dir__(self) 
   __init_subclass__(cls) 
}
class node4 {
   __init__(self, variable_file) 
   __call__(self, *args, **kwargs) 
}
class node1 {
   __hash__(self) 
}

node6  -->  node7 
object  -->  node2 
node2  -->  node6 
node7  -->  node3 
node3  -->  node9 
node8  -->  node0 
node5  -->  node10 
node10  -->  node8 
node9  -->  node5 
node1  ..>  object 
node0  -->  node4 
