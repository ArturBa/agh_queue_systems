from probability import p_local, p_online
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 


rep_dict = {0: 'Wejście', 1: 'Kelner', 2: 'Bufor Online', 3: 'Szef zup', 4: 'Czef dań głównych', 5: 'Szef deserów', 6: 'Barista', 7: 'Dostawa', 8: 'Kelner', 9: 'Kasjer', 10: 'Wyjśćie'}

# df = pd.DataFrame(p_online)
df = pd.DataFrame(p_local)
df.rename(columns=rep_dict, index=rep_dict, inplace=True)
print(df)

# sns.set_theme('notebook', 'whitegrid', 'light:blue')
f, ax = plt.subplots(figsize=(9, 6))
ax = sns.heatmap(df, annot=True, cmap="Blues")
plt.title('Tabela przejść zamówień online')
plt.show()
