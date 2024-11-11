# SmartCase
Набор инструментов для определения траектории скважин с учетом кустования с интеграцией tNavigator
![Логотип](images/smartcase.png)

## Технологии
- kmeans & GMM кластеризация для выдления зон, кустов
- networkx  для проведения графовых расчетов для определения оптимальной проводки
- plotly для визуализации итогово расчета
  
## Дополнительные материалы
1. **Исходные материалы для сбора гидродинамической модели как в примере**: [Исходные материалы для сбора гидродинамической модели как в примере](https://drive.google.com/drive/folders/1vvX-eW1IcW2JBBghcI1urRggNPUDdSY4?usp=sharing)
2. **Собранный ".exe" файл для интеграции с проектом tNavigator**: [Собранный ".exe" файл для интеграции с проектом tNavigator](https://drive.google.com/drive/folders/1KHnnp8hk4ky99qRuo7Ar4KVKqgpp0Zog?usp=sharing)
3. **Материалы результатов расчета по примеру нефтяного кейса**: [Материалы результатов расчета по примеру нефтяного кейса](https://drive.google.com/drive/folders/1Ta-AU0qcTslVdw6nYSHTIVCDJDtoED9e?usp=sharing)
5. **Материалы результатов расчета по примеру газового кейса**: [Материалы результатов расчета по примеру газового кейса](https://drive.google.com/drive/folders/1tw3A0A2y_fzMviKHpZmaCBPPzuY3rt2w?usp=sharing)

## Пример использования скриптов
1. Определение кубов OIPM, GIPM, PERMX и сетки (данную работу выполянет full.py - Workflow tNavigator)
2. Данные выгруженные кубы путем преобразований по нормализации приводятся плотностному распределению и повдение свойства будут в дальнешем описывать точки 

Примеры итоговых целевых точек. Чем ярче точки, тем выше значение расчитанного свойства

![Зоны](images/target.png)


3. Точки с нормализованными координатами и свойствами PERMX * OIPM (для нефтяного кейса) и GIPM * PERMX (для газового кейса) используются в качестве исходной информации для
   выделения ранжированных зон путем использования кластериации Kmeans. Стоит отметить что пользователь в зависимости от ситуации может выбрать другие свойства, тут нет
   особых ограничений

![property](images/wzone.png)

4. Центральные точки кластеров в данном случае зон используются для вторичной кластреизации Kmeans в целях выделения кустовых площадок.
   При определнии расположения кустов, если указан путь к растровому снимку поврехности месторождения, будут корректироваться.
   Тем самым будут исключаться ситуации возможного расположения кустовой площадки в реках, озерах и болотах.

Пример обработки .png или .jpeg  файла на предмет наличия водных объектов

![map](images/map_corr.png)


5. В дальнешйем путем использования графовых алгоритмов (алгоритм Дейкстеры, алгоритм Минимального остовного дерева - можно настроить),
   по рассматриваемиой зоне, выбираются опорные точки и линии связи с уcлоdной точкой T2 (T2 выбирается из условий буримости, углов)
   Дальше по каждому пути: [опорная точка (возмоная точка T3); линия; опорные точки; точка T2] производятся опперации сглаживания, интерполяции
   Производятся также расчеты углов, длин и расстояний, проверяются условия буримости стволов  

Пример по определению траектории скважины

![Зоны](images/wells.png)


6. Итоговые результаты расчетов траектории представляются в виде файла welltrac.INC с шагом записи 10м. В итоге после расчетов тракетории производятся расчеты для графика
   ввода скважин в виде файлов gss.xlsm и sсhedulle.sch.

Пример визуализации скважин по отношению целевых точек. Исходя из неравного мастшатба по оси z. Траектории вытянуты и видны флуктуации, но так как неровности горизональных частей скважины представлены в диапазоне от 1 до 3 метров серьезной опасности это не несет. 

![Зоны](images/wtrac.png)

## Итоговое представление таректорий скважин в проекте tNavigator

Запасы нефти

![oil](images/oil.png)


Запасы газа

![gas](images/gas.png)


Условное обозначение:
- желтые линии - газовые скважины
- зеленые линии - нефтяные скважины
- розовые линии - перфорации
  
## Порядок действий с интеграции tNavigator

01. **Необходимо установить SmartCase c Google Drive**
   
03. **Необходимо установить templeate.py**
   
04. **Необходимо установить variable.json, default_variables.json, config_varibale.json**
   
05. **Импорт Workflow templeate.py в проект Дизайнера Моделей**
   
06. **Необходимо создать директорию coord_direcotory**
   
    Директория coord_direcotory испольузется для хранения выгружаемой сетки.
    Выполняется через команду WorkFlow - "09. Выгрузка сетки"
    Абсолютный путь должен должен проходить кодировку UTF-8 (без русских букв)
   
8. **Необходимо создать директорию property_directory**
   
    Директория property_directory испольузется для хранения эскпортируемых кубов свойств сетки
    Выполняется через команду WorkFlow - "10. Сохранение кубов свойств"
    Абсолютный путь должен должен проходить кодировку UTF-8 (без русских букв)
    
10. **Необходимо создать директорию project_directory**
    
    Директория project_directory испольузется для хранения временных файлов результатов расчета
    Внутри project_directory будут создаваться директории current и history
    При повторных запусках содержимое директории current (результаты расчета после первого запуска) будет переносится в директорию history
    Абсолютный путь должен должен проходить кодировку UTF-8 (без русских букв)
    
12. **Необходимо создать директорию region_directory**

    Директория region_directory испольузется для хранения эскпортируемого дискретного куба регионов
    Выполняется через команду WorkFlow - "10. Сохранение кубов свойств"
    Данная директория должна содержать только один файл ".INC", в противном случае будет выбран первый файл
    В случае отсутсвия куба регионов, необходимо оставить папку region_directory пустым
    Абсолютный путь должен должен проходить кодировку UTF-8 (без русских букв)
    
14. **Необходимо создать директорию mask_directory**
    
    Директория mask_directory испольузется для хранения эскпортируемого бинарного (0 и 1) куба фильтра
    Выполняется через команду WorkFlow - "10. Сохранение кубов свойств"
    Данная директория должна содержать только один файл ".INC", в противном случае будет выбран первый файл
    В случае отсутсвия куба фильтра, необходимо оставить папку mask_directory пустым
    Абсолютный путь должен должен проходить кодировку UTF-8 (без русских букв)
    
16. **Необходимо создать директорию map_directory**
    
    Директория map_directory испольузется для хранения растровой карты ".png" или ".jpeg"
    Выполняется через команду WorkFlow - "10. Сохранение кубов свойств"
    Данная директория должна содержать только один файл ".png" или ".jpeg", в противном случае будет выбран первый файл
    В случае отсутсвия растровой карты, необходимо оставить папку mask_directory пустым
    Абсолютный путь должен должен проходить кодировку UTF-8 (без русских букв)

18. **Определение локальных переменных абсолютных путей к директориям и к файлам json**
    
    Выполняется через команду WorkFlow - "03. Пути к директориям" и "04. Пути к файлам json"
    
20. **Определение объекта классс Manager**
    
    Выполняется через команду WorkFlow - "15. Определение переменных моделей"
    Для оценочного запуска рекомендуется определить переменные: duration, start
    Для оптимизационной задачи рекмендуется определить переменные: duration, start, selection, wp_max, wt_max, mobil_dr
    Остальные переменные выставлены по умолчанию
    default_variables.json - необходим для сброса установленных переменных
    config_varibale.json - хранение ифнормации о переменных

Более подробная информация приведена в готовых WorkFlow templeate.py и в example.py

## Контакты
Если возникнут сложности в интеграции представленных выше решений, прошу связаться:)
Александров Николай - Aleksandrov.NA@yahoo.com - https://github.com/Nikolaytreyd
