# MatrixMultiplication

Multiplication of large square matrices using Hadoop MapReduce Java API

Каждый путь до матрицы — путь вида /path_to_matrix
(в HDFS для каждой матрицы должны создаваться следующие файл и директория: 
/path_to_matrix/data — директория с файлами, содержащими элементы матрицы;
/path_to_matrix/size — файл, содержащий размеры матрицы).

Формат файла размеров матрицы (разделитель — табуляция):
<число строк> <число столбцов>

Формат хранения элементов матрицы (разделитель — табуляция):
<тэг> <индекс i> <индекс j> <элемент Aij>

Примеры входных данных в /src/main/resources

Параметры конфигурации имеют следующие значения по умолчанию:
- mm.groups (обязательный параметр) — число групп;
- mm.tags = ABC — тэги матриц A, B, C, каждый тэг — буква латинского алфавита, все 3 тэга — разные;
- mm.float-format = %.3f — формат записи float;
- mapred.reduce.tasks = 1 — число редьюсеров.
