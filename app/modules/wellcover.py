import numpy as np

from config.importation import *
from modules.welltrac import WTrack
from modules.wellzone import WZone

class Cover (WTrack):
    def __init__(self, variable_file):
        WTrack.__init__(self, variable_file)
        self.df = None
        self._dict_wp_coords = None
        self._fwp_names = None
        self._wp_coords = None
        self._dict_wp_coords = None
        self._wp_values = None
        self._dict_wp_values = None

    def __create_date(self):
        name = []
        coords = []
        dict_coords = {}
        values = []
        dict_values = {}
        mass_1 = []  # БУ
        mass_2 = []  # КП
        mass_wp_x = []  # Координата X
        mass_wp_y = []  # Координата Y
        mass_3 = []  # Скважина
        mass_w_x = []  # Координата X
        mass_w_y = []  # Координата Y
        mass_4 = []  # Проходка общая
        mass_4_vs = []  # Проходка ВС
        mass_4_gs = []  # Проходка ГС
        mass_4_ds = []  # Проходка ДС
        mass_4_ds_num = []  # Количество доп.стволов
        mass_5 = []  # Dt мобилизации старт
        mass_6 = []  # Dt мобилизации финиш
        mass_7 = []  # Dt переезд с КП старт
        mass_8 = []  # Dt переезд с КП финиш
        mass_9 = []  # Dt переезд с скважины старт
        mass_10 = []  # Dt переезд с скважины финиш
        mass_11 = []  # Dt начала бурения
        mass_12 = []  # Dt завершение бурения
        mass_13 = []  # Dt начало ВМР, освоение
        mass_14 = []  # Dt завершение ВМР, освоение
        mass_15 = []  # Dt Готовность
        mass_16 = []  # Dt Ввод скважин
        mass_eff_trac = []
        for wp in self.get_dict_wp_wt_names.keys():
            name.append(wp)
            coords.append(self.get_dict_a_wp_points[wp])
            values.append(self.get_dict_wp_values[wp])
            dict_coords[wp] = self.get_dict_a_wp_points[wp]
            dict_values[wp] = self.get_dict_wp_values[wp]
        name = np.array(name)
        coords = np.array(coords)
        values = np.array(values)
        if coords.shape[0] <= self.mobil_dr:
            if self.auto_tuning:
                self.logger.log_warning('The number of WP is lower than the number of mobilized drilling rigs: Automatic adjustment for mobilization of 1 drill rig')
                self.mobil_dr = 1
            else:
                self.logger.log_error('The number of WP is lower than the number of mobilized drilling rigs: It is necessary to reduce the number of mobilized drill rig')
                raise AssertionError('The number of WP is lower than the number of mobilized drilling rigs: It is necessary to reduce the number of mobilized drill rig')
        labels, _, _ = WZone.Cluster.create_simple_kmeans(coords, values, self.mobil_dr, True)
        # Мобилизация старт
        moving_bu_s = self.start + timedelta(days=self.relative_mob_start)
        # Мобилизация финиш
        moving_bu_f = moving_bu_s + timedelta(days=self.mobilization_field)
        for index_bu, bu in enumerate(np.unique(labels)):
            if index_bu != 0:
                # Мобилизация старт
                moving_bu_s = self.get_relative_mob_start + timedelta(days=45 * index_bu)
                # Мобилизация финиш
                moving_bu_f = moving_bu_s + timedelta(days=self.mobilization_field)
            wp_name = name[labels == bu]
            wp_coord = coords[labels == bu]
            wp_value = values[labels == bu]
            sorted_indices = np.argsort(wp_value)[::-1]
            wp_name = wp_name[sorted_indices]
            wp_coord = wp_coord[sorted_indices]
            wp_value = wp_value[sorted_indices]
            for index_wp, wp in enumerate(wp_name):
                if index_wp == 0:
                    moving_pad_s = moving_bu_s
                    moving_pad_f = moving_pad_s
                else:
                    moving_pad_s = drill_f
                    moving_pad_f = moving_pad_s + timedelta(days=self.moving_pad)
                for index_wz, wz in enumerate(self.get_dict_wp_wt_names[wp]):
                    if index_wz == 0:
                        moving_well_s = moving_pad_s
                        moving_well_f = moving_well_s
                        if index_wp == 0:
                            drill_s = moving_bu_f
                        else:
                            drill_s = moving_pad_f
                    else:
                        moving_well_s = drill_f
                        moving_well_f = moving_well_s + timedelta(days=self.moving_w)
                        drill_s = moving_well_f
                    summary = round(sum(self.get_dict_wt_lenghts_vs[wz])) + round(
                        sum(self.get_dict_wt_lenghts_gs[wz])) + round(sum(self.get_dict_wt_lenghts_ds[wz]))
                    sinking_time = round(summary / self.passage)  # Часы бурения
                    drill_f = drill_s + timedelta(hours=sinking_time)
                    # Освоение
                    mastering_s = drill_f
                    mastering_f = mastering_s + timedelta(days=self.mastering)
                    # Готовность скважины
                    readiness = mastering_f
                    # Ввод скважины с ПМР
                    if self.start + timedelta(days=self.relative_pmr_start) >= readiness:
                        inputw = self.start + timedelta(days=self.relative_pmr_start)
                    else:
                        inputw = readiness
                    mass_1.append(f'BU-{bu + 1}')
                    mass_2.append(wp)
                    mass_3.append(wz)
                    mass_4_vs.append(round(sum(self.get_dict_wt_lenghts_vs[wz])))
                    mass_4_gs.append(round(sum(self.get_dict_wt_lenghts_gs[wz])))
                    mass_4_ds.append(round(sum(self.get_dict_wt_lenghts_ds[wz])))
                    mass_4_ds_num.append(self.get_dict_wt_quantity_ds[wz])
                    mass_4.append(round(mass_4_vs[-1] + mass_4_gs[-1], mass_4_ds[-1]))
                    mass_w_x.append(np.round(self.get_dict_wt_trajectories[wz][0][0][0], 2))
                    mass_w_y.append(np.round(self.get_dict_wt_trajectories[wz][0][0][1], 2))
                    mass_wp_x.append(np.round(self.get_dict_a_wp_points[wp][0], 2))
                    mass_wp_y.append(np.round(self.get_dict_a_wp_points[wp][1], 2))
                    mass_eff_trac.append(sum(self.get_dict_wt_values[wz]))
                    if index_wp == 0 and index_wz == 0:
                        mass_5.append(moving_bu_s)
                        mass_6.append(moving_bu_f)
                        mass_7.append(np.nan)
                        mass_8.append(np.nan)
                        mass_9.append(np.nan)
                        mass_10.append(np.nan)
                    else:
                        mass_5.append(np.nan)
                        mass_6.append(np.nan)
                        if index_wz == 0:
                            mass_7.append(moving_pad_s)
                            mass_8.append(moving_pad_f)
                            mass_9.append(np.nan)
                            mass_10.append(np.nan)
                        else:
                            mass_7.append(np.nan)
                            mass_8.append(np.nan)
                            mass_9.append(moving_well_s)
                            mass_10.append(moving_well_f)
                    mass_11.append(drill_s)
                    mass_12.append(drill_f)
                    mass_13.append(mastering_s)
                    mass_14.append(mastering_f)
                    mass_15.append(readiness)
                    mass_16.append(inputw)
        data = {
            'Бригада БУ': mass_1,
            'КП': mass_2,
            'X КП': mass_wp_x,
            'Y КП': mass_wp_y,
            'Скважина': mass_3,
            'X Скважина': mass_w_x,
            'Y Скважина': mass_w_y,
            'Проходка общая': mass_4,
            'Проходка (от устья до T2)': mass_4_vs,
            'Проходка (основной ствол)': mass_4_gs,
            'Проходка (дополнительные стволы)': mass_4_ds,
            'Количество боковых стволов': mass_4_ds_num,
            'Мобилизация (старт)': mass_5,
            'Мобилизация (финиш)': mass_6,
            'Переезд с КП (старт)': mass_7,
            'Переезд с КП (финиш)': mass_8,
            'Переезд c скважины (старт)': mass_9,
            'Переезд c скважины (финиш)': mass_10,
            'Бурение (старт)': mass_11,
            'Бурение (финиш)': mass_12,
            'Освоение (старт)': mass_13,
            'Освоение (финиш)': mass_14,
            'Готовность к эксплуатации': mass_15,
            'Ввод скважин под ПМР': mass_16,
            'Block_wells': mass_eff_trac,
        }
        self.df = pd.DataFrame(data)
        self._dict_wp_coords = dict_coords
        self._fwp_names = name
        self._wp_coords = coords
        self._dict_wp_coords = dict_coords
        self._wp_values = values
        self._dict_wp_values = dict_values
        self.logger.log_info('Determining the drilling and well commissioning schedule')

    def __save_cover(self):
        filename = f"{self.dict_path_folders['current']}//gss.xlsx"
        self.df.to_excel(filename, index=False)
        wb = Workbook()
        ws = wb.active
        for r in dataframe_to_rows(self.df, index=False, header=True):
            ws.append(r)
        thin_border = Border(bottom=Side(style='thin'))
        thick_border = Border(bottom=Side(style='thick'))
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
            if row[1].value != ws.cell(row=row[1].row - 1, column=2).value:
                for cell in ws.iter_cols(min_row=row[1].row - 1, max_row=row[1].row - 1, min_col=1,
                                         max_col=ws.max_column):
                    for c in cell:
                        c.border = thin_border
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
            if row[0].value != ws.cell(row=row[0].row - 1, column=1).value:
                for cell in ws.iter_cols(min_row=row[0].row - 1, max_row=row[0].row - 1, min_col=1,
                                         max_col=ws.max_column):
                    for c in cell:
                        c.border = thick_border
        for col in range(1, len(self.df.columns) + 1):
            cell = ws.cell(row=1, column=col)
            cell.alignment = Alignment(wrap_text=True, vertical='center', horizontal='center')
            ws.column_dimensions[cell.column_letter].width = 13  # Устанавливаем ширину столбца
            ws.row_dimensions[1].height = 50
        wb.save(filename)

    def __create_schedule(self):
        """
        Генерирует файл с записями в формате, описанном в задаче.
        """
        df = pd.read_excel(f"{self.dict_path_folders['current']}\gss.xlsx",usecols=["Скважина", "КП", "Готовность к эксплуатации", "Ввод скважин под ПМР"])
        df['Готовность к эксплуатации'] = pd.to_datetime(df['Готовность к эксплуатации'], errors='coerce')
        df['Ввод скважин под ПМР'] = pd.to_datetime(df['Ввод скважин под ПМР'], errors='coerce')

        start_date = self.start
        end_date = start_date + relativedelta(years=self.duration)
        current_date = start_date
        dates_list = []

        while current_date <= end_date:
            dates_list.append(current_date)
            current_date += relativedelta(months=1)

        with open(f"{self.dict_path_folders['current']}/well_events.INC", 'w', encoding='utf-8') as history_file:
            history_file.write("Скважина\tДата\tСобытие\tКровля\tПодошва\tГлубина\n")
            for well in self.get_dict_wt_ds_names:
                for stvol in self.get_dict_wt_ds_names[well]:
                    date = df[df['Скважина'] == well]['Готовность к эксплуатации'].values[0]
                    krovlya = self.get_dict_wtds_md_start[stvol]
                    podoshva = self.get_dict_wtds_md_end[stvol]
                    date_dt = pd.to_datetime(date).to_pydatetime()
                    date_str = date_dt.strftime('%d.%m.%Y')
                    history_file.write(f"{stvol}\t{date_str}\tperforation\t{krovlya}\t{podoshva}\tMD\n")

        with open(f"{self.dict_path_folders['current']}\schedule.sch", 'w') as file:
            for i, date in enumerate(dates_list):
                file.write(
                    f"DATES\n{date.strftime('%d')} '{date.strftime('%b').upper()}' {date.strftime('%Y')} /\n/\n\n")
                if i == 0:
                    file.write("WELSPECS\n")
                    for wt, wp in zip(df['Скважина'].values, df['КП'].values):
                        file.write(f"'{wt}'\t'{wp}'\t/\n")
                    file.write("/\n\n")
                    file.write("GRUPTREE\n")
                    file.write(f"'{self.name}'\t'FIELD'\t/\n")
                    for wp in np.unique(df['КП'].values):
                        file.write(f"'{wp}'\t'{self.name}'\t/\n")
                    file.write("/\n\n")
                    file.write("COMPDATMD\n")
                    file.write("--wname branch\tmdl\tmdu\tMD\tstatus\tfilt.tbl.\tpi\tdiameter\tkh\tskin\tD-factor mult\n")
                    for well in self.get_dict_wt_ds_names:
                        for stvol in self.get_dict_wt_ds_names[well]:
                            if ':' in stvol:
                                well_name, well_number = stvol.split(':')
                                well_number = int(well_number)
                            else:
                                well_name = stvol
                                well_number = "1*"
                            krovlya = self.get_dict_wtds_md_start[stvol]
                            podoshva = self.get_dict_wtds_md_end[stvol]
                            diameters = 0.16
                            file.write(f"'{well}' {well_number}\t{krovlya}\t{podoshva}\t1*\tPERF\t2*\t{diameters}\t3*\t1\t/\n")
                    file.write("/\n\n")
                    file.write("WELOPEN\n")
                    for well in df['Скважина'].values:
                        file.write(f"'{well}' STOP /\n")
                    file.write("/\n\n")

                if i < len(dates_list) - 1:
                    next_date = dates_list[i + 1]
                    wells_to_open = df[(df['Ввод скважин под ПМР'] >= date) & (df['Ввод скважин под ПМР'] < next_date)]
                    if not wells_to_open.empty:
                        file.write("WELOPEN\n")
                        for well in wells_to_open['Скважина']:
                            file.write(f"'{well}' OPEN /\n")
                        file.write("/\n\n")

    @property
    def get_dict_date_drill(self):
        return self.df[['Скважина', 'Готовность к эксплутации']]

    @property
    def get_dict_date_input(self):
        return self.df[['Скважина', 'Ввод скважжин под ПМР']]

    @property
    def get_fwp_names(self) -> np.ndarray:
        """
        :options:
        Возвращает актуальный спикок имен КП
        :return: ndarray
        """
        return self._fwp_names

    @property
    def get_fwp_coords(self) -> np.ndarray:
        """
        :options:
        Возвращает спикок координат кустов
        :return: ndarray
        """
        return self._wp_coords

    @property
    def get_dict_fwp_coords(self) -> Dict[str, np.ndarray]:
        """
       :options:
       Возвращает словарь имен КП и соотвествующих координат КП
       :return: Dict[str, np.ndarray]
       """
        return self._dict_wp_coords

    @property
    def get_fwp_values(self) -> np.ndarray:
        """
        :options:
        Возвращает спикок координат кустов
        :return: ndarray
        """
        return self._wp_values

    @property
    def get_dict_fwp_values(self) -> Dict[str, np.ndarray]:
        """
       :options:
       Возвращает словарь имен КП и соотвествующих координат КП
       :return: Dict[str, np.ndarray]
       """
        return self._dict_wp_values

    def __call__(self, *args, **kwargs):
        WTrack.__call__(self)
        self.__create_date()
        self.__save_cover()
        self.__create_schedule()