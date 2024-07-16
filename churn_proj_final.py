{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<!-- Projeto Desenvolvido na Data Science Academy - www.datascienceacademy.com.br -->\n",
    "# <font color='blue'>Data Science Academy</font>\n",
    "# <font color='blue'>Desenvolvimento e Deploy de Modelos de Machine Learning</font>\n",
    "## <font color='blue'>Projeto 2</font>\n",
    "### <font color='blue'>Prevendo o Churn de Clientes com RandomForest - Da Concepção do Problema ao Deploy</font>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Pacotes Python Usados no Projeto"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -q -U watermark"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-09-13T07:25:28.577903Z",
     "start_time": "2021-09-13T07:25:28.555748Z"
    }
   },
   "outputs": [],
   "source": [
    "# Imports\n",
    "import joblib\n",
    "import sklearn\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import classification_report, accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Configurações básicas para os gráficos\n",
    "sns.set(style=\"whitegrid\")\n",
    "plt.rcParams['figure.figsize'] = [6, 5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Author: Data Science Academy\n",
      "\n"
     ]
    }
   ],
   "source": [
    "%reload_ext watermark\n",
    "%watermark -a \"Data Science Academy\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Carregando os Dados"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Carregando o dataset\n",
    "df_dsa = pd.read_csv('dataset.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "pandas.core.frame.DataFrame"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Tipo de objeto\n",
    "type(df_dsa)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1000, 7)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Shape\n",
    "df_dsa.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1000 entries, 0 to 999\n",
      "Data columns (total 7 columns):\n",
      " #   Column             Non-Null Count  Dtype  \n",
      "---  ------             --------------  -----  \n",
      " 0   Idade              1000 non-null   int64  \n",
      " 1   UsoMensal          1000 non-null   int64  \n",
      " 2   Plano              1000 non-null   object \n",
      " 3   SatisfacaoCliente  1000 non-null   int64  \n",
      " 4   TempoContrato      1000 non-null   object \n",
      " 5   ValorMensal        1000 non-null   float64\n",
      " 6   Churn              1000 non-null   int64  \n",
      "dtypes: float64(1), int64(4), object(2)\n",
      "memory usage: 54.8+ KB\n"
     ]
    }
   ],
   "source": [
    "# Info\n",
    "df_dsa.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Idade</th>\n",
       "      <th>UsoMensal</th>\n",
       "      <th>Plano</th>\n",
       "      <th>SatisfacaoCliente</th>\n",
       "      <th>TempoContrato</th>\n",
       "      <th>ValorMensal</th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>56</td>\n",
       "      <td>52</td>\n",
       "      <td>Premium</td>\n",
       "      <td>1</td>\n",
       "      <td>Curto</td>\n",
       "      <td>75.48</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>69</td>\n",
       "      <td>65</td>\n",
       "      <td>Basico</td>\n",
       "      <td>4</td>\n",
       "      <td>Curto</td>\n",
       "      <td>79.25</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>46</td>\n",
       "      <td>76</td>\n",
       "      <td>Standard</td>\n",
       "      <td>3</td>\n",
       "      <td>Longo</td>\n",
       "      <td>183.56</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>32</td>\n",
       "      <td>42</td>\n",
       "      <td>Basico</td>\n",
       "      <td>2</td>\n",
       "      <td>Longo</td>\n",
       "      <td>162.50</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>60</td>\n",
       "      <td>74</td>\n",
       "      <td>Standard</td>\n",
       "      <td>2</td>\n",
       "      <td>Longo</td>\n",
       "      <td>186.23</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Idade  UsoMensal     Plano  SatisfacaoCliente TempoContrato  ValorMensal  \\\n",
       "0     56         52   Premium                  1         Curto        75.48   \n",
       "1     69         65    Basico                  4         Curto        79.25   \n",
       "2     46         76  Standard                  3         Longo       183.56   \n",
       "3     32         42    Basico                  2         Longo       162.50   \n",
       "4     60         74  Standard                  2         Longo       186.23   \n",
       "\n",
       "   Churn  \n",
       "0      0  \n",
       "1      0  \n",
       "2      0  \n",
       "3      0  \n",
       "4      1  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Amostra dos dados\n",
    "df_dsa.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Análise Exploratória"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Definimos a variável target (alvo do estudo).\n",
    "\n",
    "Variável Churn:\n",
    "\n",
    "- 1 --> Classe positiva (houve churn, ou seja, cancelou a assinatura)\n",
    "- 0 --> Classe negativa (não houve churn, ou seja, não cancelou a assinatura)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Análise Exploratória de Dados (EDA)\n",
    "def eda(dados):\n",
    "    \n",
    "    for column in dados.columns:\n",
    "        \n",
    "        # Se a coluna for numérica\n",
    "        if dados[column].dtype in ['int64', 'float64']:\n",
    "            \n",
    "            # Histograma e Boxplot\n",
    "            fig, axes = plt.subplots(1, 2)\n",
    "            sns.histplot(dados[column], kde = True, ax = axes[0])\n",
    "            sns.boxplot(x = 'Churn', y = column, data = dados, ax = axes[1])\n",
    "            axes[0].set_title(f'Distribuição de {column}')\n",
    "            axes[1].set_title(f'{column} vs Churn')\n",
    "            plt.tight_layout()\n",
    "            plt.show()\n",
    "            \n",
    "        # Se a coluna for categórica\n",
    "        else:\n",
    "            \n",
    "            # Contagem de frequência e relação com Churn\n",
    "            fig, axes = plt.subplots(1, 2)\n",
    "            sns.countplot(x = column, data = dados, ax = axes[0])\n",
    "            sns.countplot(x = column, hue = 'Churn', data = dados, ax = axes[1])\n",
    "            axes[0].set_title(f'Distribuição de {column}')\n",
    "            axes[1].set_title(f'{column} vs Churn')\n",
    "            plt.tight_layout()\n",
    "            plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkcAAAHkCAYAAAA0I4sqAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABsH0lEQVR4nO3deVxU5f4H8M8MM8MMOyKLOwSJoiKaKJYr7ZZ1udStfsJVMtM0vWWmlVZYmXozzeVaqbhXmuFVM1u97ZEKbpmooYgLIsgOMzDb+f2BjDOCMIzDbHzer1ev5Jwzz/nOGZj5zHOe8xyRIAgCiIiIiAgAILZ3AURERESOhOGIiIiIyAjDEREREZERhiMiIiIiIwxHREREREYYjoiIiIiMMBwRERERGWE4IiIiIjLCcNRGce5PIiKixjEcOaDk5GRERkYa/uvRowf69euHv//979i0aRN0Op3J9vHx8XjppZfMbn/v3r2YNWtWs9u99NJLiI+Pt3g/N7Jv3z5ERkZi3759Zj8mMzMTw4cPR48ePRAVFYWoqCiMGzfupmu5keufu6WWL1+OyMjIZrez57Elak38G2hIr9dj27ZtGDNmDAYNGoT+/fsjISEBGzduhFqtNmx34cIFREZGYvv27Xastm2S2LsAalxUVBRef/11AIBOp0N5eTl+/PFHvP3228jKysKSJUsgEokAACtWrICXl5fZba9fv96s7SZPnox//vOfLa69Ob169cLWrVsRERFh9mNCQ0Px4YcfQq1WQyqVQi6Xo1u3blavjYioNalUKkyaNAlHjhzBE088gaeeegpSqRT79u3DokWL8OOPP+L999+HTCazd6ltGsORg/Ly8kJMTIzJsvj4eISFhWH+/PmIj4/HQw89BKAuSLWGrl27tkq7jT235rRv3x7t27dvlXqIiGxl/vz5OHjwIDZt2mTyPjhkyBBERUXhueeew0cffYSUlBT7FUk8reZskpOTERQUhC1bthiWXd8dvWfPHjz00EOIjo5GXFwcZsyYgcLCQsPj9+/fj/379xu6nuu7obds2YKRI0fi9ttvxy+//NLoqSWNRoO33noLsbGxiI2NxaxZs1BSUmJY39hjru8abqzb++jRoxg/fjxuu+02xMXFYfr06bh8+bJh/YkTJ/Dss88iLi4OvXr1wtChQ/HWW2+hpqbGsE1tbS3+85//4L777kOfPn1wzz33YNWqVdDr9U0e0/Lycrz88ssYNGgQYmNj8c477zT6mO+++w5///vf0adPH9xxxx146623oFQqm2y7MSdOnEBKSgr69euHkSNHYteuXQ22KSkpwdy5czFy5Ej07t0bAwcOxJQpU3DhwgWT7bZs2YJ7770X0dHRSEpKQn5+foO28vPzMX36dAwcOBB9+/bF2LFjcfz48RbXTWQtjv43cPDgQURGRuK7774zWX769GlERkbiyy+/BND0e21jSkpKkJ6ejsTExEa/IN5///0YP348QkJCTJYXFRVh2rRp6NevHwYOHIhXX33V5L2nsVOS27dvR2RkpOF4LV++HHfffTdWrFiBQYMG4a677kJpaSni4+OxbNkyLFy4ELfffjuio6Mxfvx45Obm3vB5tAXsOXIybm5uGDx4MPbs2QOtVguJxPQlzMrKwowZMzB58mTExsaioKAA77zzDl544QVs2rQJr7/+Ol588UUAwOuvv46IiAj8+eefAIAlS5Zg7ty5qK2tRUxMDHbv3t1g/19++SWio6OxYMEClJSUYNGiRcjLyzMJay114sQJjBkzBtHR0Vi4cCF0Oh3effddjB8/Hjt27EBJSQnGjBmDmJgYLFiwADKZDD/88AM2bNiA9u3bY9KkSRAEAZMmTcLhw4cxZcoU9OzZE/v27cN7772H8+fP480332x033q9Hk899RQuXLiAGTNmICAgAGvWrMHRo0cRFBRk2O7zzz/HjBkzMHr0aDz33HO4ePEilixZgpycHKxbt85wirM5ly9fRlJSErp27Yp33nkHVVVVWLRoEYqLiw3bCIKAiRMnory8HC+88AICAwORnZ2NpUuX4rXXXsPatWsBAJs3b8abb76J5ORkjBgxAhkZGXj11VdN9ldSUoLHH38cCoUCr776KhQKBTZs2IAxY8bgs88+Q3h4eEtfLqKb4gx/A/3790e3bt2wZ88e3HXXXYbln3/+Oby9vREfH9/se21jMjIyoNVqMXLkyBsen5kzZzZYtnTpUiQnJ2PlypXIysrC8uXL4eXlZdbYUWP5+fn49ttvsXjxYpSWlsLf3x8AsHHjRtx2222YP38+ysvLMW/ePLz00kvYunVri9p3JQxHTqh9+/bQaDQoKytrcKopKysL7u7umDBhAtzd3QEAfn5++OOPPyAIAiIiIgzjk67/5vL444/j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",
      "text/plain": [
       "<Figure size 600x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 600x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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GLlcjkePHmHWrFlYt25drS6uJCIiaiicViONycrKwgcffACZTAZfX99W9UwlIiJqujitRkRERKSE02pEREREShiOiIiIiJQwHBEREREp4QXZT/jjjz8giuJT701BRLWjUCggCAJ69+6t6aY0GxyLiOpfTccihqMniKLYKh+yR9SQ+J6qPY5FRPWvpu8phqMnVHxKq+3DEYmoeufPn9d0E5odjkVE9a+mYxGvOSIiIiJSwnBEREREpIThiIiIiEgJwxERERGREoYjIiIiIiX8thoREVELVVZWBoVCoelmNAodHR1oaWnVy7YYjoiIiFoYURSRnZ2N+/fva7opjapdu3awsrKCIAh12g7DERERUQtTEYwsLCxgaGhY57DQ1ImiiMLCQuTm5gIArK2t67Q9hiMiIqIWpKysTApGZmZmmm5OozEwMAAA5ObmwsLCok5TbLwgm4iIqAWpuMbI0NBQwy1pfBXHXNfrrDQejvLy8hAaGgoPDw/07t0bH374IdLT06X6uXPnwsHBQeVn4MCBUn15eTmioqLg6ekJV1dXjB8/HpmZmZo4FCIioiajpU+lVaW+jlnj4eijjz7C9evXERMTg2+++Qb6+voYN24cioqKAACXL19GUFAQjh8/Lv18++230vobNmzAnj17sHjxYsTHx0MQBAQGBkIul2voiIiIiKg502g4unfvHjp16oRFixahZ8+esLW1xeTJk3H79m389ddfKCsrQ3p6Onr27Alzc3Ppx9TUFAAgl8uxdetWTJs2DV5eXnB0dMTq1auRk5ODhIQETR4aERFRs3L+/HmEhoZi0KBBcHFxwZAhQzB//nxcv35dWiYgIAABAQEabGXj0Gg4at++PSIjI9G9e3cAwJ07dxAbGwsrKyvY2dnh6tWrKCkpga2tbZXrp6am4tGjR/Dw8JDKTExM4OTkhMTExEY5BiIiouZu165dGD16NPLy8hASEoKYmBgEBQUhMTERfn5+uHjxoqab2KiazLfVPvnkE+zduxe6urrYuHEjDA0NkZaWBkEQEBcXh2PHjkEmk8HLywvBwcEwNjZGdnY2gMpf2bOwsEBWVpbaban4SiAR1Q9RFFvl9Q9EzUFycjKWLFmCMWPGICwsTCrv27cvhgwZgrfeegtz587FgQMHNNjKxtVkwtHYsWMxatQofPXVV5gyZQp2796Nv/76CzKZDB07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KffJN998g8WLF2PevHlwd3eHKIo4deoUlixZgjt37mDq1Kn4/PPPcePGDYwcOVLTzaVmgGOR+jgWcSx6FoajWjI0NIS5ubn02sLCAuHh4Rg4cCD+/e9/4/3332+UdrRp0wZt2rRplH3Vlyf7rnPnzjh9+jT27dtXLwNJU+6T3bt34+2338Y777wjlT3//PPIzs7Gjh07MHXqVPB+rFQbHIvUx7GIY9GzcFqtHmhrP86Yurq68Pb2xtKlS/Haa6+hb9+++O233yCKImJiYjBkyBC4urrijTfewIEDB6T1T58+DScnJ/z222947bXX0LNnT4waNQoZGRnYuHEj/vGPf+Cll17CokWLpD9a5dO2N27cgIODg8oDLwHAwcEB+/fvl5YfN24cduzYgQEDBqBXr16YMWMGbt++jVmzZqF3797w8vLCv/71r8boMomBgYH0e0FBARYsWAAvLy/06NED/fv3x4IFC1BcXCwtExsbi5dffhnOzs7w9vbGF198UWWfAMDdu3cxe/Zs9O3bF3369EFgYCCuXr0q1R85cgTvvPMOevfujQEDBmD58uUoKSlpkOOUyWT4/fffkZ+fr1IeGBiI+Ph4zJkzB//6179w5swZODg41Kg/Tp8+DQcHBxw9ehTDhw+Hs7Mzhg0bhv/85z/S9uVyOZYuXYp+/frBzc0Nq1atQnl5uUobfv31V4wePRq9e/dGz5498fbbb+PkyZNSfUBAAObNm4eRI0fCzc0N3377LURRxIYNGzBw4ED06tULYWFhDdZ3VHMci9THsYhjkQqRamzw4MFiVFSUSll2drY4ffp0sVevXuLNmzfFwYMHi87OzuKJEyfEc+fOiSUlJeKqVavEQYMGib/++quYmZkpfvPNN2Lv3r3FL7/8UhRFUfztt99Ee3t78Y033hD/+OMP8dKlS+KQIUPEl156SQwJCRHT09PFr776SrS3txd//fVXURRFMSoqShw8eLAoiqJ4/fp10d7eXvztt99U2mZvby/u27dPWr5Hjx7itGnTxPT0dPE///mP6OTkJL700kvi1q1bxStXrojz588XnZ2dxbt37zZK3/3555+ih4eHuGfPHlEURTEoKEh88803xbNnz4rXr18Xv//+e9HZ2Vncvn27KIqiePjwYdHNzU08fvy4ePPmTfGHH34Qe/ToIX777beV+kShUIhvvPGG+Oab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",
      "text/plain": [
       "<Figure size 600x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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M2BWW9y+fRtt2rRBeHh4seX//PNPqS05vr6+SEhIQGxsrHxZWFgYAMDb27tqgyWiKsPkiIjoBScnJ7Rs2RILFixAeHg47t27hyVLluDatWsYN24cCgoKkJSUJP9l7enpCW9vb0ydOhU3btzApUuXMG/ePPTt25fT+Im0GJMjIqIXxGIxNm7ciNatW2PKlCno168frl+/jh9++AHOzs6Ij49HQEAAjh8/DqCwiX7t2rWwt7fHiBEjMGXKFAQGBmL+/PmaPRAiqhSOOSIiUmBlZYXFixeXus7e3h7R0dHFlllbW2P16tXVERoRVRO2HBEREREpYHJEREREpIDJEREREZECJkdERERECpgcERERESlgckRERESkgMkRERERkQImR0REREQKmBwRERERKWByRERERKSAyRERERGRAiZHRERERAqYHBEREREpYHJEREREpIDJEREREZECJkdERERECpgcERERESlgclQJgiDUqHqIiIio8nQ1HYA2E4lEuHAjDunP81Suo56pAV73aKTGqGoXQRAgEolqTD1ERFT7MTmqpPTneUjNVD05oldjAkpERNVN48lRWloali9fjlOnTuH58+dwdnbG9OnT4ePjAwCIiorCokWL8M8//8DCwgLDhg3DqFGj5NvLZDKsXbsWP/30EzIyMtC2bVvMmzcPzZo109QhkZoxASUiouqk8TFH06ZNw/Xr17F8+XL8/PPPcHNzw6hRo3Dv3j2kpqbigw8+QPPmzXHgwAFMnDgRq1atwoEDB+Tbr1+/Hnv37sXChQuxb98+iEQijBkzBhKJRINHRURERNpKoy1HsbGxOH/+PH788Ud4e3sDAObMmYMzZ87g2LFjMDQ0hL6+PubPnw9dXV04OjoiNjYWmzdvRv/+/SGRSLB161Z8+umn6NSpEwBgxYoV6NixI0JDQ/HWW29p8vCIiIhIC2m05cjS0hKbNm2Cu7u7fJlIJIIgCEhPT0d4eDh8fX2hq/tfDufv748HDx4gOTkZt2/fRlZWFvz9/eXrzc3N4erqiitXrlTrsRAREVHtoNGWI3Nzc3mLT5HffvsN//77LwICArBixQo4OTkVW9+gQQMAQFxcHBISEgAADRs2LFEmPj5e5bgEQUB2dvYry4hEIhgZGSE/Px9SqVTlfeXn6wAAcnJyqn1Kf05OTrH/a5racI5L8/JxFR1bRY9R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",
      "text/plain": [
       "<Figure size 600x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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N0tlU6dORXx8vMp8jRo1wuHDhwE8uXM2KioKO3fuRHZ2Njp27IiZM2eiRYsW1Vp3InpxNSIcbdmyBeHh4fD19UVERIR0hKlQKKRgVEKpVAIAUlNTpctp6enpaN68uTRPeno6nJycNK6PQqGAiYmJxsu/DOrKJSBNsX3KV1bb1KZwuHHjRkRGRsLDw0OlnHfOEtUdWn/O0bZt2zB79mwMGTIES5cuVelEJk6ciBEjRqjMf/HiRQCAg4MDnJycYGpqipMnT0rTs7OzcenSJbi7u1fPDhDRSyEtLQ0ffPABli1bBjs7O5VpvHOWqG7RajhKTk7G3Llz4evri+DgYGRkZODOnTu4c+cOHjx4gL59++LYsWNYtWoVrl+/jp9//hnTpk1D3759YW9vDwMDAwQEBCAiIgI//vgjLl++jPHjx8PW1ha+vr7a3DUiqmV+//13WFhYYO/evWjXrp3KNN45S1S3aPWy2vfff4+CggIkJCSUOrry9/fH/PnzsWzZMqxevRqrV6+GmZkZ+vXrh3HjxknzjR07FoWFhZgxYwZyc3Ph4eGBmJgYnsYmogrx8fGBj49PmdN45+w/NLmTtbppq33YNuWrKW2j7p2zWg1HISEhCAkJeeY8fn5+8PPzK3e6rq4uQkNDERoaWtnVIyICAFy9epV3zv5/mtzJWt201T5sm/LVpLZR5+RJjRiQTURUk/HO2X9ocidrddNW+7BtyldT2kbdO2cZjoiInoN3zv6Dd2qWj21TvprSNuqGQ63frUZEVNPxzlmiuoXhiIjoOXjnLFHdwstqRETP0bNnT945S1SHMBwRET1l/vz5pcp45yxR3cHLakREREQyDEdEREREMgxHRERERDIMR0REREQyDEdEREREMgxHRERERDIMR0REREQyDEdEREREMgxHRERERDIMR0REREQyDEdEREREMgxHRERERDIMR0REREQyDEdEREREMgxHRERERDIMR0REREQyDEdEREREMgxHRERERDIMR0RERC/AwswIori42rerjW3WFXrargAREVFtVs/IAAodHSTvi0ZOxu1q2aaxVWPY9Q2qlm3VRQxHRERElSAn4zZy0q5ruxpUCXhZjYiIiEiG4YiIiIhIhuGIiIiISIbhiIiIiEiG4YiIiIhIhuGIiIiISIbhiIiInksbDzrkQw5JW/icIyIieq7qftAhH3JI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/ODn51fuMnwTNhEREVWmGn1ZjYiIiKi6MRwRERERyTAcEREREckwHBERERHJMBwRERERyTAcEREREckwHBERERHJMBwRERERyTAcEREREckwHBERERHJMBwRERERyTAcEREREckwHBERERHJMBwRERERyTAcEREREckwHBERERHJMBwRERERyTAcEREREckwHBERERHJMBwRERERyTAcEREREckwHBERERHJMBwRERERyTAcEREREckwHBERERHJMBwRERERyTAcEREREckwHBERERHJMBwRERERyTAcEREREckwHBERERHJMBwRERERyTAcEREREckwHBERERHJMBwRERERyTAcEREREckwHBERERHJMBwRERERyTAcEREREckwHBERERHJMBwRERERyTAcEREREcnUqHC0cuVKDB06VKUsMTERAQEBcHNzQ48ePRATE6Myvbi4GJGRkfDy8kK7du0QGBiIlJSU6qw2ERERvURqTDjauHEjIiMjVcoyMzMxfPhwtGzZErt27cKYMWOwbNky7Nq1S5pn5cqV2L59O+bMmYO4uDgoFAoEBQUhPz+/uneBiIiIXgJ62q5AWloapk+fjrNnz8LOzk5l2o4dO2BgYICwsDDo6enB3t4eKSkpiI6ORv/+/ZGfn4/169cjNDQU3t7eAIAlS5bAy8sLCQkJ6NOnjzZ2iYiIiGoxrZ85+v3332FhYYG9e/eiXbt2KtPOnDkDDw8P6On9k+E8PT2RnJyMjIwMXL58GY8ePYKnp6c03dzcHM7Ozjh9+nS17QMRERG9PLR+5sjHxwc+Pj5lTktNTYVSqVQps7GxAQDcunULqampAIDGjRuXmuf27dsa10kIgcePH5cqVygUMDY21ni9tUlOTg6EEBVahu1TvrreNkIIKBQKLdWIiKhitB6OniU3NxcGBgYqZYaGhgCAvLw85OTkAECZ82RlZWm83YKCAiQmJpYqNzY2hrOzs8brrU2Sk5Ol9lUX26d8bJvSn1MiopqqRocjIyOjUgOr8/LyAAAmJiYwMjICAOTn50vfl8zzIkfp+vr6cHBwKFVel4587ezsNDpzVFdUtH3qetskJSVpqTZERBVXo8ORra0t0tPTVcpKfm7UqBEKCwulsubNm6vM4+TkpPF2FQoFTExMNF7+ZVBXLgFpiu1TvrLapi6FQyKq/bQ+IPtZPDw8cPbsWRQVFUllJ06cgJ2dHaysrODk5ARTU1OcPHlSmp6dnY1Lly7B3d1dG1UmIiKiWq5Gh6P+/fvj4cOHmD59OpKSkhAfH4/Y2FgEBwcDeDKGISAgABEREfjxxx9x+fJljB8/Hra2tvD19dVy7YmIiKg2qtGX1aysrLBu3TqEh4fD398f1tbWmDRpEvz9/aV5xo4di8LCQsyYMQO5ubnw8PBATEwMB38SERGRRmpUOJo/f36pMldXV8TFxZW7jK6uLkJDQxEaGlqVVSMiIqI6okZfViMiIiKqbgxHRERERDIMR0REREQyDEdEREREMgxHRERERDIMR0REREQyDEdEREREMgxHRERqunnzJhwdHUt97dy5EwCQmJiIgIAAuLm5oUePHoiJidFyjYlIEzXqIZBERDXZH3/8AUNDQxw4cEDlZbpmZmbIzMzE8OHD8dprr2HWrFk4f/48Zs2ahfr166N///5arDURVRTDERGRmq5cuQI7OzvY2NiUmhYbGwsDAwOEhYVBT08P9vb2SElJQXR0NMMRUS3Dy2pERGr6448/4ODgUOa0M2fOwMPDA3p6/xxzenp6Ijk5GRkZGdVVRSKqBAxHRERqunLlCjIyMjB48GC88sorGDRoEI4cOQIASE1Nha2trcr8JWeYbt26Ve11JSLN8bIaEZEa8vPz8ddff8HY2BiTJk2CiYkJ9u7di6CgIGzYsAG5ubkwMDBQWcbQ0BAAkJeXp9E2hRB4/PhxqXKFQgFjY2ON1lnb5OTkQAhRoWXYPuWr620jhFAZL1gehiMiIjUYGBjg9OnT0NPTk0JQ27Zt8eeffyImJgZGRkbIz89XWaYkFJmYmGi0zYKCAiQmJpYqNzY2hrOzs0brrG2Sk5ORk5NToWXYPuVj26DUQUxZGI6IiNRUVshRKpU4evQobG1tkZ6erjKt5OdGjRpptD19ff0yxzipc+T7srCzs9PozFFdUdH2qettk5SUpNayDEdERGq4fPkyBg0ahOjoaLi7u0vlv/32GxwcHNC6dWts374dRUVF0NXVBQCcOHECdnZ2sLKy0mibCoVC47NOL4u6cglIU2yf8pXVNuqGQw7IJiJSg1KpRKtWrTBr1iycOXMGf/75J+bNm4fz588jJCQE/fv3x8OHDzF9+nQkJSUhPj4esbGxCA4O1nbViaiCeOaIiEgNOjo6WL16NSIiIjBu3DhkZ2fD2dkZGzZsgKOjIwBg3bp1CA8Ph7+/P6ytrTFp0iT4+/trueZEVFEMR0REarK0tMTcuXPLne7q6oq4uLhqrBERVQVeViMiIiKSYTgiIiIikmE4IiIiIpJhOCIiIiKSYTgiIiIikmE4IiIiIpJhOCIiIiKSYTgiIiIikmE4IiIiIpJhOCIiIiKSYTgiIiIikmE4IiIiIpJhOCIiIiKSYTgiIiIikmE4IiIiIpJhOCIiIiKSYTgiIiIikmE4IiIiIpJhOCIiIiKSYTgiIiIikmE4IiIiIpJhOCIiIiKSqRXh6ObNm3B0dCz1tXPnTgBAYmIiAgIC4Obmhh49eiAmJkbLNSYiIqLaSk/bFVDHH3/8AUNDQxw4cAAKhUIqNzMzQ2ZmJoYPH47XXnsNs2bNwvnz5zFr1izUr18f/fv312KtiYiIqDaqFeHoypUrsLOzg42NTalpsbGxMDAwQFhYGPT09GBvb4+UlBRER0czHBEREVGF1YrLan/88QccHBzKnHbmzBl4eHhAT++fnOfp6Ynk5GRkZGRUVxWJiIjoJVErwtGVK1eQkZGBwYMH45VXXsGgQYNw5MgRAEBqaipsbW1V5i85w3Tr1q1qrysRERHVbjX+slp+fj7++usvGBsbY9KkSTAxMcHevXsRFBSEDRs2IDc3FwYGBirLGBoaAgDy8vI02qYQAo8fPy5VrlAoYGxsrNE6a5ucnBwIISq0DNunfHW9bYQQKuMFiYhqshofjgwMDHD69Gno6elJIaht27b4888/ERMTAyMjI+Tn56ssUxKKTExMNNpmQUEBEhMTS5UbGxvD2dlZo3XWNsnJycjJyanQMmyf8rFtUOoghoiopqrx4QgoO+QolUocPXoUtra2SE9PV5lW8nOjRo002p6+vn6ZY5zq0pGvnZ2dRmeO6oqKtk9db5ukpCQt1YaIqOJqfDi6fPkyBg0ahOjoaLi7u0vlv/32GxwcHNC6dWts374dRUVF0NXVBQCcOHECdnZ2sLKy0mibCoVC47NOL4u6cglIU2yf8pXVNnUpHBJR7VfjB2QrlUq0atUKs2bNwpkzZ/Dnn39i3rx5OH/+PEJCQtC/f388fPgQ06dPR1JSEuLj4xEbG4vg4GBtV52IiIhqoRp/5khHRwerV69GREQExo0bh+zsbDg7O2PDhg1wdHQEAKxbtw7h4eHw9/eHtbU1Jk2aBH9/fy3XnIiIiGqjGh+OAMDS0hJz584td7qrqyvi4uKqsUZERET0sqrxl9WIiIiIqhPDEREREZEMwxERERGRDMMRERERkQzDEREREZEMwxERERGRDMMRERERkQzDEREREZEMwxERERGRDMMRERERkQzDEREREZEMwxERERGRDMMRERERkQzDEREREZEMwxERERGRDMMRERERkQzDEREREZEMwxERERGRDMMRERERkQzDEREREZEMwxERERGRDMMRERERkQzDEREREZEMwxERERGRDMMRERERkQzDEREREZEMwxERERGRDMMRERERkQzDEREREZEMwxERERGRDMMRERERkQzDEREREZEMwxERERGRDMMRERERkQzDEREREZEMwxERERGRDMMRERERkQzDEREREZEMwxERERGRDMMRERERkQzDEREREZEMwxERERGRzEsRjoqLixEZGQkvLy+0a9cOgYGBSElJ0Xa1iKiOYV9E9HJ4KcLRypUrsX37dsyZMwdxcXFQKBQICgpCfn6+tqtGRHUI+yKil0OtD0f5+flYv349xowZA29vbzg5OWHJkiVIS0tDQkKCtqtHRHUE+yKil0etD0eXL1/Go0eP4OnpKZWZm5vD2dkZp0+f1mLNiKguYV9E9PLQ03YFXlRqaioAoHHjxirlNjY2uH37doXXV1BQACEEfv311zKnKxQK9OlkjaJiq4pXthbQ1dHBxYsXIYTQaHmFQoFCp9egUBZVcs1qhjwdXY3bR6FQwK9xNxQ2ejnbRu8ZbVNQUACFQqGFWlWfl70vMtDXw8WLF6vt8/0inzWgetunutsGePG+qK62jbp9Ua0PRzk5OQAAAwMDlXJDQ0NkZWVVeH0ljfasxjM3NarwemubF/lHpmdiVok1qZk0bR8zI9NKrknNU1bbKBSKlz4c1ZW+qLo/3y/yd1Pd7aONvk/T9qmrbaNuX1Trw5GR0ZNfcH5+vvQ9AOTl5cHY2LjC62vfvn2l1Y2I6g72RUQvj1o/5qjkFHZ6erpKeXp6OmxtbbVRJSKqg9gXEb08an04cnJygqmpKU6ePCmVZWdn49KlS3B3d9dizYioLmFfRPTyqPWX1QwMDBAQEICIiAhYWlqiSZMmWLhwIWxtbeHr66vt6hFRHcG+iOjlUevDEQCMHTsWhYWFmDFjBnJzc+Hh4YGY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",
      "text/plain": [
       "<Figure size 600x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkcAAAHkCAYAAAA0I4sqAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAACCWklEQVR4nO3deVhU1f8H8PcMzLBvIou7JgmighuIKaKYaVr2RW35FWTupmmZ+5JLZWpa7ua+W5pppqaVWmklKZLmBhqm5sIWssMw2/39Mc40IyAwDMwMvF/P05Pce+bcz70zc+9nzjn3XJEgCAKIiIiICAAgNncARERERJaEyRERERGRHiZHRERERHqYHBERERHpYXJEREREpIfJEREREZEeJkdEREREepgcEREREelhcmQhOBcnERGRZWByVA4xMTHw9/fX/RcQEIB27dphwIAB2LFjB1QqlUH5yMhITJs2rdz1nzhxAlOnTi2z3LRp0xAZGWn0dkpz5swZ+Pv748yZM+V+zblz5xAREYGAgAAEBgYiMDAQb7zxRqVjKc2j+26MIUOGIDQ0FHK5vNQyL7zwAl588cVy1RcTE4OYmJhKxaSl/Wx9+umnJa5Xq9UIDw+Hv78/9u/fb5JtmpoxnyOyDPxuVC1zfzdSU1Px8ccfo0+fPggODkbXrl0xatQoxMXFGZRbuXIl/P39zRKjpbE1dwDWIjAwEHPmzAEAqFQqZGdn4+TJk/joo48QHx+PpUuXQiQSAQBWrVoFZ2fncte9devWcpUbM2YMXn/99QrHXpZWrVphz5498PPzK/drmjZtinXr1kEul0MikcDe3h5NmjQxeWymNGjQIJw+fRqnTp3C008/XWx9YmIiEhMT8f7775shOkAsFuO7777Du+++W2xdXFwc0tLSzBAV1Qb8btRc8fHxGDt2LDw8PPD666+jWbNmyM7OxpdffomYmBh8+OGHGDRokLnDtDhMjsrJ2dkZbdu2NVgWGRmJZs2aYcGCBYiMjET//v0BaBKpqtC4ceMqqbekfStL3bp1Ubdu3SqJp6r06tULbm5uOHjwYIkXgAMHDsDR0RH9+vUzQ3RA+/btce7cOVy5cgWtWrUyWPftt9+iZcuWSEhIMEtsVLPxu1EzZWVl4Z133kHTpk2xZcsWODg46NY988wzGDNmDObNm4du3brB29vbjJFaHnarVVJMTAy8vb2xe/du3bJHu7uOHDmC/v37IygoCGFhYZg0aZLul05MTAzOnj2Ls2fP6ppdtU2wu3fvRo8ePfDUU0/h119/LbFrSaFQ4MMPP0RISAhCQkIwdepUPHjwQLe+pNfcvXvXoAm6pCbfixcvYtiwYejQoQPCwsLw7rvvIjU1Vbc+MTERb731FsLCwtCqVSuEh4fjww8/hEwm05UpKirC6tWr0adPH7Rp0wbPPPMM1q9fD7Va/dhjmp2djenTp6NTp04ICQnB4sWLS3zN8ePHMWDAALRp0wZdunTBhx9+iIKCglLrlUqleP755/HTTz8hNzfXYJ1KpcLhw4fRp08fODs748GDB5g3bx569OiB1q1bIzQ0FGPHjsXdu3dLrb88+xsTE4NJkyZh/PjxaN++PUaOHKlbFxoairp16+Lo0aMG9SqVSvzwww8lXpiysrIwe/ZsPPXUU2jTpg1eeuklxMbGGpTx9/fHrl27MHPmTISGhqJdu3YYP348/v33X12ZO3fu4M0330SnTp0QHByMl19+GSdPnix2vF999VW0a9cOrVu3Rp8+fbBz585SjwdZD343zPvdeO+99xAWFgalUmmwfPHixbruzqKiIl0io93G5s2bH1vvgQMHkJaWhhkzZhgkRoCmNW7ixIl47bXXkJeXZ7Du559/Rv/+/dGmTRv07t0bBw4c0K0rrYvw0a7UyMhIfPTRRxg8eDDat2+P2bNn614bGxuLoUOHIjg4GE899RQWLVpUbN/NjclRJdnY2KBz5864ePFiiW9ufHw8Jk2ahGeeeQYbNmzA9OnT8fvvv2PixIkAgDlz5ujG7OzZs8fgV9HSpUsxdepUTJ06tdSWnaNHj+Ly5ctYuHAhpkyZgp9//hljxoyp1D4lJibitddeg0wmw6JFizBv3jxcvnwZw4YNg1KpRFpaGl577TUUFhZi4cKF2LBhA5599lns2LFD10UoCAJGjx6NjRs3YtCgQVi7di369OmDZcuW6bonS6JWqzF8+HD8/PPPmDRpEhYtWoTz58/jyJEjBuUOHTqEsWPH4oknnsDq1avx1ltv4eDBgxgzZsxjB7cPGjQIcrkc3333ncHyX3/9Fenp6Rg0aBAEQcCoUaPw22+/YeLEidi0aRPGjBmD06dPY/bs2SXWW5H9PXr0KCQSCVavXm3QTSoWi9G7d+9iscXGxqKoqAg9evQwWF5UVITBgwfjxIkTmDBhAlatWgVfX18MHz682EVg6dKlUKvV+PTTT3Wfk48++kh3zEeNGoWCggJ8/PHHWLNmDdzd3TFmzBjcvn0bgOZkOXbsWLRq1Qpr1qzBypUr0aBBA3zwwQf4448/Sj3eZD343TDfd+OFF15AZmamQWyCIODIkSPo06cPpFIp5s+fj5MnT2Lq1KnYtGkTevbsiUWLFj12nNUvv/wCT09PBAUFlbj+ySefxLRp0/DEE08YLJ89ezbeeOMNfPbZZ/D29sa0adOQmJhYrn3Rt2vXLvj7+2PlypV44YUXdMsnTZqEDh06YO3atXj++eexefNmfPXVVxWuvyqxW80E6tatC4VCgaysrGJdTfHx8bCzs8OIESNgZ2cHAHB3d8elS5cgCAL8/Px045MeTYBeeeUV9OnT57HbdnV1xcaNG3V1eHh4YOzYsfj111/RtWtXo/ZnzZo1cHNzw+bNm3Uxe3t7Y+LEifjrr7+QkZGBli1bYvny5brtPvXUU4iNjUVcXBxGjx6NU6dO4fTp01i8eLGuu7FLly6wt7fH8uXLMXjw4BLHOJ06dQoXL17EunXr0L17dwBAWFiYQeuXIAhYsmQJwsPDsWTJEt3ypk2b4o033sDJkyd1r31Uy5YtERgYiEOHDhkMLv3666/RvHlzdOjQAampqXBwcMDUqVPRsWNHAECnTp1w9+5dgxbCR+Mu7/6KxWJ88MEHcHR0LFZP3759sWvXLly+fBmtW7cGoGl57NmzJ+zt7Q3KfvPNN0hMTMSXX36J4OBgAEC3bt0QExODJUuWYN++fbqyLVq0wIIFC3R/X7x4UXehycjIwI0bNzB69GhEREQAAIKCgrBq1SoUFRUBAJKSkvC///0PM2fO1NXRrl07dOrUCXFxcWjfvn2Jx4WsB78bGub4bnTo0AENGzbEkSNHEB4eDkBz7bh//74uqTh79iyeeuopXStZp06d4OjoCA8Pj1LrTU1NRcOGDcvc/qM+/PBDdOvWDQDQqFEjPPPMMzh79iwCAgIqVI82sRKLNe0w2tamF198EWPHjgUAdO7cGcePH8fPP/+MV155pcKxVhW2HJmQdkC2vpCQEMhkMjz//PNYunQp4uPj0bVrV7z11lslltdXnrsGIiIiDAZ/R0ZGQiKR4PTp0xXfgYfi4+PRrVs3XWIEaL7sP/74I1q2bImuXbti586dsLOzw82bN/HTTz9h7dq1ePDgge5ul7Nnz8LGxgZ9+/Y1qFt7ciztro1z585BIpHovpgA4OjoqDsxAcDff/+NlJQUREZGQqlU6v4LCQmBs7Mzfvvtt8fu36BBgxAXF4eUlBQAQG5uLn788UfdoEQfHx9s374dHTt2xP379xEbG4udO3fijz/+gEKhKLHOiuxvw4YNSzz5A5qTpI+Pj677QC6X4/jx43juueeKlY2NjYWXlxdatWqlOwYqlQo9evTA5cuXkZ2drSv7aOLt6+uLwsJCAJrk3s/PD++99x6mTZuGI0eOQBAETJ8+HS1atAAADB8+HIsWLUJBQQESExNx9OhRrF+/HgBKPSZkffjdMM93QyQSoX///jh27JjuHHr48GE0atQIHTp0AKBJhvbu3YsRI0bg888/x7179zB27NhirWaP1vvo3dTloU18AU1yBAA5OTkVrqd58+a6xEhfu3btDP729fV97JAIc2ByZAKpqamwt7eHu7t7sXXt2rXD+vXr0ahRI2zatAmvvvoqIiIisG3btjLr9fT0LLPMoy1VYrEY7u7uRn2QtbKysh67bbVajSVLliA0NBR9+vTBvHnzcPXqVYNkKjs7Gx4eHrC1NWyc9PLyAoBi4xr0X+fu7l7sC6V9nTY+AJg3bx5atWpl8F9eXl6Zd648//zzkEgkOHz4MADNr0+1Wm3Q7Hvw4EF0794dPXr0wDvvvINjx44V+3X6aNzl3d/HDWQXiUTo06eP7pfrL7/8ArFYjC5duhQrm5WVhfT09GLH4OOPPwYApKen68qWNN5A2/0oEomwefNmREVF4ZdffsGECRPw1FNP4Z133tEd6wcPHmDcuHHo0KEDBgwYgBUrVug+Y5yjq+bgd8N8343//e9/yM3NxalTp6BUKvHdd9/pEkgAmDlzJt555x3cvXsX8+bNQ2RkJF555RVcvXq11DobNGiA5OTkx263pPX6Car2XGzM97y09/PRz4v+MbcU7FarJJVKhbNnz6J9+/awsbEpsUx4eDjCw8NRWFiI33//Hdu3b8dHH32Etm3b6pp8jfVoEqRSqZCZmalLbkr65VBWhu7i4mIwqFvr5MmTaNmyJfbv34+tW7di7ty56N27N1xcXADA4HZQNzc3ZGZmQqlUGpwUtYlLaU3BHh4eyMzMhEqlMjie2hMRoOlKBIApU6YgNDS0WB1ubm6P3T9XV1f06tULhw4dwvDhw3HgwAFERkbqjtm5c+cwdepUREdHY9iwYfD19QUAfPzxx4iPjy+xTmP3tyR9+/bFtm3bcOnSJRw5cgTPPPMMJBJJsXIuLi5o2rSpQdeivoo0p/v4+GDu3LmYM2cOEhMT8d1332HDhg1wc3PDvHnzMGnSJNy4cQNbtmxB+/btIZVKUVhYiL1795Z7G2T5+N0orrq+G02aNEHbtm11464yMzMNkiOpVIo333wTb775Ju7fv4+ffvoJa9aswcSJE4sNVNcKDw/HTz/9hEuXLqFNmzbF1v/111947rnnMHHiRIMB8I+j7fF49CaZ/Px8ODk5lXd3LR5bjipp9+7dSEtLw//93/+VuH7RokW6gYwODg7o0aOHbsJHbcZeUrNjeZ0+fdpgIPj3338PpVKJTp06AQCcnJyQmZmp6x8HUOYgwY4dO+KXX34xmBDu6tWrGDlyJC5fvoz4+Hj4+flh0KBBusQoNTUV169f131hQkNDoVKpig2kPnjwIADomoof1blzZyiVShw/fly3TC6XG3SVPfHEE/D09MTdu3fRpk0b3X++vr745JNPHvtLSmvQoEFITEzE2bNncf78eYPE7vz581Cr1Rg/frzu5K9SqXRdlSXdOWfs/pakbdu2aNCgAQ4dOoQff/yx1NunQ0NDkZycDE9PT4PjEBsbi40bN5aarD/q/PnzeOqpp3Dx4kWIRCK0bNkSEyZMQIsWLXTdK/Hx8ejduzfCwsIglUoBaMaSACUfD7Je/G78p7q/G/3798epU6dw+PBhtG3bFk2bNgUAyGQy9O7dW3d3Wv369fHaa6+hX79+ujhKq8/LywsfffSRrqtQS61WY/HixZBIJBWaokE7jEO/xSk7Oxs3btwodx3WgC1H5ZSXl4cLFy4A0HyoMjMz8euvv2LPnj3o378/nnnmmRJf17lzZ2zZsgXTpk1D//79oVAosHHjRri7uyMsLAyA5tfa+fPnERsbW+E5kv7991+MGzcOMTExuHXrFj799FN06dIFnTt3BgD06NEDO3bswIwZM/Diiy/ir7/+wubNmx97chgzZgxefvlljBgxAm+88QZkMhmWLVuG1q1bo2vXrrhy5QrWrFmD9evXo23btrh9+7ZuQkjtF7Bbt27o1KkT5syZg7S0NAQGBuLs2bPYsGEDoqKiSp1wsnPnzujatStmzZqFjIwMNGjQANu3b8eDBw90v15tbGwwYcIEzJ49GzY2NujRowdycnKwZs0apKamFpsHpSRhYWFo2LAh3nvvPfj6+hoMXtfe2fH+++9j4MCByMnJwc6dO3V3axQUFBSb5NPY/S1Nnz59sH37dri7u5fYOgYAAwYMwM6dOzFkyBCMHj0a9erVw+nTp7FhwwZER0eX+Iu6JIGBgbC3t8eUKVMwbtw41K1bF6dPn0ZCQoLujqGgoCAcOnQIrVq1gq+vL86fP49169ZBJBIVO+mSdeN34z/V/d3o168fFixYgG+//dZggLe9vT1atWqFVatWQSKRwN/fHzdv3sTXX3+N3r17l1qfi4sLFi5ciLfeegsvvvgioqOj0axZM6SkpOCLL77AhQsXsHDhQjRo0KDcMfr7+6NevXpYtWoVXFxcIBaLsX79+mLdk9aOyVE5Xb16FS+//DIATUuPp6cnmjVrhoULF+L5558v9XXdunXDkiVLsHnzZt0g7A4dOui+3ADw2muv4fLlyxgxYgQWLFhQocm4XnrpJchkMowdO1Y3V8nkyZN1TZ9dunTB1KlTsWPHDvzwww+6L9jj7goIDAzEjh078Mknn2D06NGQSqV47rnnMGnSJEilUowaNQqZmZnYvn07Vq9ejXr16uGFF16ASCTCunXrkJ2dDTc3N6xbtw4rVqzQJTcNGzbEhAkTMGTIkMfu06pVq7BkyRKsWLECRUVF6Nu3L1566SWcOHFCV+bFF1+Ek5MTNm7ciD179sDR0RHt27fHkiVLdAMIH0ckEunGB4wdO9ag9a5Tp06YPXs2tmzZgu+++w5169ZFp06dsGrVKowdOxbx8fEGA8S19Rm7vyXp27cvNm3ahGeffbbUlkVHR0fs2rULn3zyCRYvXozc3Fw0aNAAEydOxNChQ8u9LTs7O2zevBmffPIJ5s+fj5ycHDRt2hTvv/8+BgwYAABYuHAhPvjgA3zwwQcANHcGzps3DwcPHsS5c+cqvH9kufjd+E91fzfc3d0RERGBkydPFhvA/v7772PZsmXYvHkz0tPT4enpiUGDBuHtt99+bJ1du3bF3r17sXnzZmzYsAHp6elwc3NDq1at8MUXXxQbHF0WGxsbrFixAh999BHeffdd1K1bF4MHD8bff/+NmzdvVqguSyYSLG0UFFmMv/76C4MGDcKIESPw5ptvlrspmoiIyJpxzBGVSC6XIz8/H1OmTMHKlStLHWxJRERU07BbjUqUnJyMIUOGQCwWIyoqqsLPXiMiIrJW7FYjIiIi0sNuNSIiIiI9TI6IiIiI9DA5IiIiItLDAdmPOH/+PARBKPckYURUOoVCAZFIVOG5VOg/PCcRmU55z0lsOXqEIAjlegCeIAiQy+UW97C86lLb9x/gMSjP/pf3+0Sl4zEkMp3yfp/YcvQI7a+zkh7Sp6+goAAJCQnw8/MzeIJxbVHb9x/gMSjP/l+6dKmao6p5yntOIqKylfecxJYjIiIiIj1MjoiIiIj0MDkiIiIi0sPkiIiIiEgPkyMiIiIiPUyOiIiIiPQwOSIiIiLSw+SIiIiISA+TIyIiIiI9TI6IiIiI9DA5IiIiItLD5IiISE9WVhZmz56Nbt26oX379vi///s/nDt3Trc+ISEB0dHRaNu2Lbp3745NmzYZvF6tVmPFihUIDw9HcHAwhg4ditu3b1f3bhBRJTA5IiLS8+677+LPP//Ep59+iq+++gqtWrXCsGHDcOPGDWRmZmLIkCFo2rQp9u3bh3HjxmH58uXYt2+f7vVr1qzB7t278eGHH2LPnj0QiUQYMWIE5HK5GfeKiCrC1twBEBFZitu3b+O3337DF198gfbt2wMAZs6ciVOnTuHw4cOwt7eHVCrF3LlzYWtri+bNm+P27dvYsGEDBg4cCLlcjs2bN2Py5MmIiIgAACxduhTh4eE4duwY+vXrZ87dI6JyYnJERPSQh4cH1q9fj9atW+uWiUQiCIKA7OxsXL58GSEhIbC1/e/UGRYWhnXr1iEjIwP37t1Dfn4+wsLCdOtdXV0RGBiIuLg4JkdUZQRBQFFRkcnrBDTfAVOys7MzeZ2mxuSIiOghV1dXXYuP1tGjR/HPP/+ga9euWLp0KVq0aGGw3tvbGwBw//59pKSkAADq1atXrExycrLRcQmCgIKCAqNfXxm86Fo+QRAwe/ZsXL9+3dyhlIu/vz/mzZtnlvdKEIRybZfJERFRKeLj4zFjxgz07NkTkZGRWLBgAaRSqUEZOzs7AEBRUREKCwsBoMQy2dnZRsehUCiQkJBg9OuNJQgCNm/ejDt37lT7to3RqFEjDB06tNYlSIIg6D571qCgoAAJCQlme58e/X6WhMkR1Wjl/ZVQXfWQ9Th+/DgmTZqE4OBgfPrppwAAe3v7YgOrta0qjo6OsLe3BwDI5XLdv7VlHBwcjI5FIpHAz8/P6NcbSxAEODo6Vvt2jeXo6IiWLVvWyu/q4sWLTdrCV1RUhJEjRwIA1q9fr/sRYArmbOFLSkoqVzkmR1SjiUQinL54H9l5xp803Jzt8FRQfRNGRZZu586dmD9/Pnr16oUlS5bofmn6+voiLS3NoKz2bx8fHyiVSt2yxo0bG5QJCAgwOh6RSGS2JMXUF12ZTIaYmBgAwI4dOwySyMqqrd1qWk5OTiarSyaT6f7t4eFh0vfJnMr7+WByRDVedl4RMnNNO2aCaq7PP/8cH3zwAWJiYjBjxgyIxf/NeBISEoLdu3dDpVLBxsYGABAbG4tmzZrB09MTLi4ucHZ2xpkzZ3TJUU5ODq5evYro6Giz7E9liUSiKrsw2tvb15iLLtUsnOeIiOihmzdv4qOPPkKvXr0watQoZGRkID09Henp6cjNzcXAgQORl5eHmTNnIikpCfv378e2bdswatQoAJqxDNHR0ViyZAlOnDiBxMRETJgwAb6+vujVq5eZ946IyostR0RED33//fdQKBQ4duwYjh07ZrAuKioKCxcuxMaNGzF//nxERUXBy8sLU6ZMQVRUlK7c+PHjoVQqMWvWLMhkMoSEhGDTpk3lGgRKRJaByRER0UOjR4/G6NGjH1smKCgIe/bsKXW9jY0NJk+ejMmTJ5s6PCKqJuxWIyIiItLD5IiIiIhID5MjIiIiIj1MjoiIiIj0MDkiIiIi0sPkiIiIiEgPkyMiIiIiPUyOiIiIiPQwOSIiIiLSw+SIiIiISA+TIyIiIiI9TI6IiIiI9DA5IiIiItLD5IiIiIhID5MjIiIiIj1MjoiIiIj0MDkiIiIi0sPkiIiIiEgPkyMiIiIiPUyOiIiIiPRYVHK0Zs0axMTEGCxLSEhAdHQ02rZti+7du2PTpk0G69VqNVasWIHw8HAEBwdj6NChuH37dnWGTURERDWIxSRHW7duxYoVKwyWZWZmYsiQIWjatCn27duHcePGYfny5di3b5+uzJo1a7B79258+OGH2LNnD0QiEUaMGAG5XF7du0BEREQ1gK25A0hNTcXMmTMRHx+PZs2aGaz78ssvIZVKMXfuXNja2qJ58+a4ffs2NmzYgIEDB0Iul2Pz5s2YPHkyIiIiAABLly5FeHg4jh07hn79+pljl4iIiMiKmb3l6MqVK3Bzc8PBgwcRHBxssO7cuXMICQmBre1/OVxYWBhu3ryJjIwMJCYmIj8/H2FhYbr1rq6uCAwMRFxcXLXtAxEREdUcZm85ioyMRGRkZInrUlJS0KJFC4Nl3t7eAID79+8jJSUFAFCvXr1iZZKTk42OSRAEFBQUPLZMYWGhwf9rG2vYf5FIBAcHByiVSigUCqPrUSptAGj2VRAE3XJrOAZVqTz7LwgCRCJRdYVERGQSZk+OHkcmk0EqlRoss7OzAwAUFRXpTsollcnOzjZ6uwqFAgkJCeUqe+vWLaO3UxNY8v47ODggMDAQmVmZSM/IM7oekdoZAHDz5s0SEwFLPgbVoaz9f/T7SURk6Sw6ObK3ty82sLqoqAgA4OjoCHt7ewCAXC7X/VtbxsHBwejtSiQS+Pn5PbZMYWEhbt26haZNm1ZqW9bKGvZf22Lh4e4BQWx8jB6ums9Ws2bNirUcWfoxqErl2f+kpKRqjoqIqPIsOjny9fVFWlqawTLt3z4+PlAqlbpljRs3NigTEBBg9HZFIhEcHR3LVdbBwaHcZWsia9h/W1tbSCSSSr0eQKkJgDUcg6r0uP1nlxoRWSOzD8h+nJCQEMTHx0OlUumWxcbGolmzZvD09ERAQACcnZ1x5swZ3fqcnBxcvXoVHTt2NEfIREREZOUsOjkaOHAg8vLyMHPmTCQlJWH//v3Ytm0bRo0aBUAzliE6OhpLlizBiRMnkJiYiAkTJsDX1xe9evUyc/RERERkjSy6W83T0xMbN27E/PnzERUVBS8vL0yZMgVRUVG6MuPHj4dSqcSsWbMgk8kQEhKCTZs2cRAoERERGcWikqOFCxcWWxYUFIQ9e/aU+hobGxtMnjwZkydPrsrQSiSRSDimgoiIqIaxqOTImohEIrRq1Qo2NjaVqofzwBAREVkWJkeVYGNjg1Pn7yC/UGnU692c7fBUUH0TR0VERESVweSokrJzZcgtVJVdkIiIiKyCRd+tRkRERFTdmBwREZVizZo1iImJMVh26dIlREdHo127doiIiMDHH39sMJO/Wq3GihUrEB4ejuDgYAwdOhS3b9+u7tCJqBKYHBE9IidfjoRbD3AuIRUXrqfj5v1sKJRqc4dF1Wzr1q1YsWKFwbIHDx5g+PDheOKJJ3DgwAF88MEH+Prrr7F06VJdmTVr1mD37t348MMPsWfPHohEIowYMaLYo5CIyHJxzBHRQ0UKFf5ITMOt5ByD5Qm3HiDuaiqGvdAaXYMbmCk6qi6pqamYOXMm4uPj0axZM4N1f/zxB7KysjBlyhQ4OzujSZMm6N+/P3799VdMnToVcrkcmzdvxuTJkxEREQEAWLp0KcLDw3Hs2DH069fPHLtERBXElqMaQv+BqOasw1rlFsjx/e+3dYmRr6cjWj3hCb+G7nC0t8W/2TIs2n4Oa/dfhFpde49TbXDlyhW4ubnh4MGDCA4ONljn7u4OAPjiiy+gUqlw9+5dnDx5UlcuMTER+fn5CAsL073G1dUVgYGBiIuLq7Z9IKLKYctRDSESiXD64n1k5xUZ9fraPK1AfqECJ+LuoLBICWcHCZ4KqgdPt/8eMuvi2BBZeXLsPXEd3/52EzK5EuNeamfGiKkqRUZGIjIyssR1HTt2xMiRI7F8+XIsXboUKpUKoaGheO+99wAAKSkpAIB69eoZvM7b2xvJyclGxyQIAgoKCox+vSWRyWS6fxcUFECtZpe1Jaqp71N55xZkclSDZOcVITPXuOSotlKq1Dh5/i4Ki5RwdZIismMjONgZfi1sbcSIebYlmvq6Ysnn8TgRdwc2YjGG9nvSTFGTueTk5ODWrVt47bXX0L9/f9y5cwcLFizA3LlzsWDBAhQWFgJAsccX2dnZITs72+jtKhQKJCQkVCp2S6E/9uratWt81JOFqsnvU3n2hckR1Wp/JKYhO08Oe6kNundoWCwx0hfergEgApbsiscPZ26jfl17NHWrxmDJ7JYsWYKcnBysXLkSANCqVSu4ubnhjTfewODBg2Fvbw9Ac2HR/hsAioqK4ODgUGKd5SGRSODn51e54C2EfouEv7+/wXEiy1FT36ekpKRylWNyRLVWSkY+btzT/Jp/Kqg+nOwlZb4mvG0DZGTLsOngZWw/eh2v96iLli2rOlJDpnjkDB9bY5z4+Hj06NHDYJl2vNHNmzfRoIFmwH5aWhoaN26sK5OWloaAgACjtysSieDo6Gj06y2JWPzfUFdHR8cac9GtaWrq+1Te8x6TIzKgVgtQqNQQizQfIhuxqEZeRJUqNeKupgIAnmzkDp865b/wvNDtCSTdycLJ83fx1ekMhIcqUJ3XLY4vMx9fX19cu3bNYNn169cBAE2bNkXz5s3h7OyMM2fO6JKjnJwcXL16FdHR0dUeLxEZh8lRLVQkV+LfbBmycouQWyBHXoECMrkS2769ioIiJfRvWrO1EcPD1Q51XOzh4+mI5g3c4N+kDhp72ZlvB0zgyt8ZyCtUwMHOFsFP1q3Qa0UiEd56MRh/3XmA+/8WYPO3iZg2uFMVRVoyji8zjyFDhmD48OFYtmwZBgwYgHv37mHevHmIiIhAy4dNiNHR0ViyZAnq1KmDBg0aYPHixfD19UWvXr3MHD0RlReTo1pApVYj9UEh7qfnITkjH3kFinK/VqlSIz2zEOmZhbj2TyZOnb8HAHC0t4WfrxRRDpno0NLBqlqXCmQKXLudCQDoEOANia1Nheuwt7PFmAGt8d6Gs/jtYgp++/M+ugSzNaam69q1K9atW4fVq1dj27Zt8PDwQK9evfD222/ryowfPx5KpRKzZs2CTCZDSEgINm3aVKMGtBLVdEyOaihBEPAgR4aku9n4JyUHSpXh3DwujlJ4utnDxVECZ0cpfD0d0Su0CZwdJbCT2EAQHt4+LFPiQa4MmTky3EnNw417WbiUlIHcAjku3lLi4qZzaN7QDa8/G4h2/l5WkSRdupEBlVqAl7sDGno7G13Pk43c0DXQBb9cycXqr/5EYLM68HCtGf3ypLFw4cJiyyIiInQTPJbExsYGkydPxuTJk6syNCKqQkyOahi1IOBOSi6u3nqALL1uFwc7G9T3ckaDus7w8nCAVGLYWuLhYodGPi7F6nN2lML74Xiczm00y1RqARevJ+PAj1dx5Y4MN+5mY86GWHRs6YO3Xgw2mCPI0mTnFeHmw0HYbVtUPpmLaO2KOxnArRRNgjRzSKhVJIhERFQ6zpBdQwiCgKS7WTh6+hZOX0pGVm4RxGIRmtZzRc+QRnihW3OEBvqigbdzscSoomzEIgQ08cDL3etj9cRw/C+iOWxtxDiXkIq3Fv+E2EvGT3ZX1S7fyIAAoIGXM+q6Vz6Js7URYezAVrC1EeHMlRScfNjtSERE1ovJUQ1wJzUXMz87jR/O/IOcfDkktmK0ae6J/0U0R+c29eDt4Vhma4a91KZCjw9xcHBAYGAgfOq6YVj/1lj+bgT8Grohr1CBj7aexVc//mVxjyPJzZfjn9RcAECQX8UGYT9OY18XvNzLHwCw/uuLyMyVlfEKIiKyZOxWs2JqtYCDv9zAtm+vQqkSYGsjQkCTOvBv4lHh1iGpxKZCt4grlUpkZmXCw90Dtraaj1HPkMZwc05FfGIatn17FRlZhRgZ1cZiupkSbj0AANSv6wR3F9PebTco8knEXkzG3/ez8dm+i5g+OMRi9puIiCqGyZGVyiuQY/GuePyRmAYA6NjSBwFNPKCq5ENRy3uLuEKhQHpGHgSxAySS/yZP7NTKFx0CfLDhm0s4/NtNiG1EGN6/tdkThQKZAjfva8YaBTarY/L6bW3EePuVdnh32UnEXkrGr3/eR3jbBibfDhERVT12q1mhlIx8TF75C/5ITIPUVowxg4Ixe1gnuDpZxq3Cz4c/gbdebAsAOHjqb3z+/bXHv6AaXLudCbUAeLk7wMujamZsfKKBG17s2QIAsHb/RaMnaSQiIvNicmRl7qTmYvLKX3A3LQ913eyx5O1ueLZzU7O3zDzqmU5N8ObAIADA7mPX8HP8HbPFIpMrkXQ3C0DVtBrpe+npFmhazxU5+XKs+/pSlW6LiIiqBpMjK5L8bz5mrT2NrNwiNK3niiVvd0Oz+pb75NO+TzXDwB6ah2Uu33MBibcfmCWOyzcyoFQJcHOWol5dpyrdlsRWjLdfbgexWIRfLtzDb3/er9LtERGR6TE5shLZeUWYte40HuTI0NjXBR+Ofsqi5xPSer1vIMJa+0KpUuPjHeeQVyCv1u3L5EpcTPoXABDYzNOkLWwikQgODsVnB/dr5K5LCld+eR4pGfkm2yYREVU9JkdWQKUWsGRXPNIeFKCepxM+HPUU3Jwt89lmj04JIBaLMOH/2qOepxPSMwux4ssL5b7F3xRTARw78w9kchWcHCRoXMIkl+VR2jQH2ukMHByKJ6mv9g6AfxMP5MuUWLTjHBRKlcVNbUBERCXj3WpW4PPvE3HhejrspDaYOSTUoh9RUdqUAF2C62H/TzcQeykZS3bGo3Vzz8fWY4onxytVanx9MgkA0LKpB8Ri41qNStunkqYz0Bca6INbyTlIupOFOetjMf/NLsbtCBERVSsmRxYu8dYD7D1xHQAw7sW2aFLP1cwRlc+jUwJIbG0Q3KIuzl9Lx68X78PJQWLyuYYeder8XaRnFsLBztYkY7Me3afSpjPQF9a6Hk6dv4tLNzKw67tERD/bstJxEBFR1WK3mgVTqtRYtfcCBAGI7NgIEe0bmjukSvFv7IH6dZ2gVguIvZQMlVpdZdtSqQV8efwvAJrZsG1tzPNRr1/XCSEtfQAAe45fx94T19m9RkRk4ZgcWbCvf07C7ZRcuDhKMfT5VuYOp9JEIhFCW/nCTmKDrLwiXL6RUWXbOn3xPu6l58HJQYI2ZXThVbXmDd0RGqhJkLYfScD6A5egUlVdYkhERJXD5MhCZebKsOe4pjtt+AutLXYAdkU52Nki5GGikHDzAdKzCk2+DbVawJcPj13/8Ccq/aBdU+jY0gcjXmgNkQg4/OtNTF75C24l55g7LCIiKgGTIwu178ckFMlVaNHYHT06WHd32qMa+bigaT1XCAB+v5QMhdK0rShnr6bgVnIOHOxs8Xz4EyatuzL6d2uOqTEhcHKQ4K87WXj705+xcFsczl9Lg6xIae7wiIjoIQ7ItkAZ2YU4evomAOC1Pi0tbvZrU+gQ4I20zALkFSpw4Xq6rjWpsgRB0LW49evSDC6OlvFIFa0uwfUR0NQD676+hNhLyfjt4n38dvE+xGIR6td1gquTFPZ2thBB0w0pFokgEgEiEeDhao8WjdwR5Odl7t0gIqrRmBxZoK9+/AtypRotm9ZBuxY180IoldigUytf/BR/F0l3s9DQ29kks1fHJ6Yh6U4WpBIb/C+iuQkiNT1PNwfMeCMUt5JzcOS3mzhzJQUPcmS4m5ZX5muPPvx/Yx8XtGxWB3UseFoHIiJrxeTIwuQVyPHDmX8AAK/1DqiRrUZavp5OaNHYHdf/ycKZKyno+1TTSo0PUqkFbD18BQDQ96mmFj9Oq2k9V4wZFIwxg4KRnlmI++l5yCtUQCZXQhA0rWACNP9XqwUkZxQg8dYDJN5+gH9Sc3EnNReBT3iidXNPiGvw54SIqLoxOTIj7czL+gnQiXN3IFeo0LSeK4KerGvG6KpH8JNeSP43H7kFCpxLSK3UxI/Hz97G7ZRcODtI8NLTLUwYZdXz8nCAl0f5HgeT/G8+Pt4Rh6S72bjydway84rwVFB92Bg5ySURERlicmRGj868LAgCvjqhmZunsa8Lvou9Va566ns5I/hJ6+x+s7URo3Obejh29h/cTslFQ+8cNPat+ESXBTIFdn6XCAB45Rl/ixtrZEr16jrhmU5N4OWRjjNXUnA3LQ+xl+6jS1D9Gt3SSERUXZgcWQDtzMspGfnIyiuCrY0I3h6OBrMxP46rk3UnAp5uDghs5okrf2cgLiEVnm4O8Kjg7NnbjyQgK7cI9eo6oe9TzaooUsvStJ4r7CQ2OHX+Hu6k5uHSjQwE+dX81kYioqrGW/ktSNLdbACai57Etna9Na2f8EQdVzvIFWqcunCvQrf3X7iehm9/09zd9+aAoFp17OrVdUJoK82dflf+zkDqgwIzR0REZP1qz1XEwimUKtxL19yt1Lyhu3mDMQOxWISuwQ00s2fnFuHHc3egUpf9mI3cAjmW7z4PQHPrfjt/76oO1eI0q++G5g00z447cyUFSs6+TURUKUyOLMS99Hyo1QJcHCUV7lKqKZwcJAhvWx9iEXDjXjaW7f7jsY/ZUChVWLgtDv9my1CvrhPe6BdYjdFalnb+3nC0t0V+oQJX/q66x7IQEdUGTI4sxD8puQCAxr6utXpQrZeHIzq3qQeRCPg5/i4W7TiHApmiWLkCmQIfbjmLi0n/wsHOFtMHh8DervYOoZPYitEhQNNqdu12Jgo54zYRkdFq79XEgsgVKiRn5AMAGvk4mzka82vs6wpXJymOnb2D2EvJuHE3Cy/38kdIoA/EIhHOX0vDru8TkZJRADupDWa+EYpm9d3MHbbZNfByhqebPTKyZbh684EuWSIioophcmQBbiXnPOxSk8LdwicurC7N6rth/puNsGRXPNIyC7HyywvFynh5OGBqTEf4N6lT/QFaIJFIhCC/uppZx+9kIaCpB5zsJeYOi4jI6rBbzQLcuKe5S62xr0ut7lJ7VGAzT6yZEok3+gU+PDaa5fXrOuHV3gFYNakHE6NH+NRxhJeHA9SCgMRbD8wdDhGRVWLLkZkplGrdM7UaerNL7VH2UlsMjHwSAyOfhEKpAgBIbI1/xIi5lDQbelUQiURo1cwTP2fexc37OQjy86pVUxvUVoIgoKiofPOimZtMJivx35bMzs6OP1xrGSZHZnbt9gMolGrYSWxq7V1q5WWNSZHWo7OhG6s8s6H7ejrC2VGCvAIFbqfkwK8WTg1R2xQVFeHFF180dxgVFhMTY+4QymXv3r2wt+dDnmsTJkdmdv56OgDNBY2/TGo+7WzoxirPbOgikQh+Dd1x4Xo6/rqTheYN3PjZIiKqACZHZvbHtTQAmifUE5nKE/XdcCnpX2TlFiEjW4a67uV7qG1FiEQiSCQc8G1pnJ78H0Riyz61C4JmgldLTtoFtRL5fx0wdxhkJpb9DarhCmQK3LibBYDJEZmWndQGjX1dcPN+Dv6+l22QHJlq/JODgwNatWoFuVxe2XDJhERiW4tPjiw3JSLSsOxvUA13OzkXggDUcbWHoz3fCjKtpvVccfN+Du6k5aJjSx+IxZpLkqnGPzk52KJbu0amCpeIyGLwimxGN5NzAHDiR6oa3h6OsJPYoEihQuqDAtSra9g6WdnxT0olZ+EmopqJ9/ia0Z1UzSNDGnoxOSLTE4tFaOTjAgC4nZJj5mis05o1a4rdUZWWloZ3330XHTt2RKdOnTBx4kQ8ePDfnFJqtRorVqxAeHg4goODMXToUNy+fbu6QyeiSmByZCayIqXuV7uPp6OZo6GaqrGvJjm6m5YHlVowczTWZevWrVixYoXBMrlcjqFDh+LOnTvYsmUL1q1bh6tXr2Lq1Km6MmvWrMHu3bvx4YcfYs+ePRCJRBgxYgTHZhFZESZHZvJvdiEAzcXLXsreTaoaXh4OsJfaQKFUI+Xh8/vo8VJTUzF8+HAsX74czZo1M1h3+PBh3Lt3D5999hnatGmDtm3bYsaMGbh58yby8vIgl8uxefNmjBs3DhEREQgICMDSpUuRmpqKY8eOmWmPiKiimByZSXqWJjlq2ZSPv6CqIxb917V2Lz3PzNFYhytXrsDNzQ0HDx5EcHCwwbpffvkFYWFhqFu3rm5ZeHg4jh8/DmdnZyQmJiI/Px9hYWG69a6urggMDERcXFy17QMRVQ6bLMzkX73kSK5QmTkaqsnq13XCX3eykPxvvm5+GSpdZGQkIiMjS1x369YtdOzYEatXr8aBAwegVCrRtWtXTJ48Ga6urkhJSQEA1KtXz+B13t7eSE5ONjomQRBQUFBQ6npreQyHtSooKIBarTZ3GNVK/zNVk/a/vNOYWEVypFAosGrVKnzzzTfIzs5Gy5YtMWnSJLRv3x4AkJCQgPnz5+Py5ctwd3dHTEwMhg0bZuaoS6dSq/EgWzPeqGXTOvjzr3QzR0Q1mXcdR4jFIhTIlMjJ57iXysjLy8OBAwfQuXNnfPLJJ8jOzsaCBQswZswY7NixA4WFmh89UqnhTOZ2dnbIzs42ersKhQIJCQmlrud4pqp17dq1Yu9pTaf/mapp+1+efbGK5Oizzz7Dvn37sHDhQjRq1AgbNmzAiBEjcOTIEUilUgwZMgRPP/005s2bhwsXLmDevHlwd3fHwIEDzR16iR7kFEEtCHC0t0W9uk5MjqhK2dqI4e3hgJSMAtz/Nx9BfnXLfhGVSCKRwNHREZ988oludnA3Nze8+OKLuHTpku75W3K53OBZXEVFRXBwMH6WcolEAj8/v1LXs+Woavn7+9e6Z6vpf6Zq0v4nJSWVq5xVJEcnTpzAc889h65duwIApk2bhr179+LChQu4desWpFIp5s6dC1tbWzRv3hy3b9/Ghg0bLDY50napNfBytujp86nmqFfXCSkZBUj+l4OyK8PX1xdqtdrgsSlPPvkkAODu3bto2LAhAM3t/o0bN9aVSUtLQ0BAgNHbFYlEcHQs/a5WsZjDR6uSo6NjjUkOykv/M1WT9r+811yr+Ea5u7vjp59+wt27d6FSqbBnzx5IpVK0bNkS586dQ0hICGxt/8vzwsLCcPPmTWRkZJgx6tJl6JIjPjKEqkf9upq5tNIzCznGrRI6duyIxMREg1/V169fBwA0adIEAQEBcHZ2xpkzZ3Trc3JycPXqVXTs2LHa4yUi41hFcjRz5kzY2tqiZ8+eaNOmDZYuXYply5ahcePGSElJga+vr0F5b29vAMD9+/fNEW6ZHjyc3+jRGYuJqoqLowRODhKoBQG3U3LNHY7VeuWVV2BjY4OJEyfi+vXriI+Px6xZs9CpUye0atUKUqkU0dHRWLJkCU6cOIHExERMmDABvr6+6NWrl7nDJ6JysoputRs3bsDV1RWrV6+Gj48P9u7di6lTp2Lnzp2QyWQlDn4ENP38xijrzhBAM6bAwcEBSqUSCkX5H6MgV6iQX6gAANR11cSpqUNhVKwAoFKpKl1PRevQlnm0rCli0bzeBgBQWFho9B1WIpFI7z0y/fEt7RhUpA5TxVIe3h72uFmowK3kbJPEopRo3peioqJS3yNTPODWktSpUwe7du3CggUL8NJLL0EqleLpp5/G9OnTdWXGjx8PpVKJWbNmQSaTISQkBJs2bapRA1qJajqLT47u3buHyZMnY+vWrbpm6TZt2iApKQkrV66Evb19sTs1tEnR4/roH6esO0MAzRPJ3d3dkZuXi/SM8s8fk5mnSaTsJSLI5ZoELDcvF+npWUbFCgCeLqJK12NsHVlZhmVNEQsAiNSabqCbN2/q7gCqKAcHBwQGBiIzK7NC79GjytqnR4+BMXWYKpbHsRNrEqHbD5/pV+n3yFPzHt2/f/+x75E1JwULFy4stqxp06ZYt25dqa+xsbHB5MmTMXny5KoMjYiqkMUnRxcvXoRCoUCbNm0MlgcHB+PUqVOoX78+0tLSDNZp//bx8TFqm2XdGQL8d5uji7MLBHH570LJLMwCUAhPd0e4urjq6vBSSx77uscxRT0VrUOhUCArKwvu7u4Gg1NNtU8erprBf82aNatUyxEAeLh7VOg9elRp+1TaMahIHaaKpTycXJRIvPsPMnKKUCBTVDoWF0fN6aN+/fqlJkDlvTOEiMiSWHxypJ1M7dq1awgKCtItv379Opo0aYK2bdti9+7dUKlUsLHRdMXExsaiWbNm8PT0NGqbZd0Zoi0DALa2tpBIyt9tkJ2v+fXu6eagi1dTh/EXKVPUY2wdEonEoLyp9kk7wL4ytz/r11WVx/fRY2BMHaaK5XHcJBI4O0iQV6hAwq0HJniPNLHY2dmV+j7VpC41Iqo9LH5AdlBQEDp27IipU6fi999/x61bt7Bs2TLExsZi5MiRGDhwIPLy8jBz5kwkJSVh//792LZtG0aNGmXu0Eukfdish4udmSOh2sjLQ5PEXL5hmXdyEhFZAotPjsRiMdasWYOwsDBMnz4dAwYMwO+//46tW7eibdu28PT0xMaNG3Hz5k1ERUVh1apVmDJlCqKioswdejEKpVo3Q7G224ioOnl7aFpEr/zN5IiIqDQW360GaGagnTNnDubMmVPi+qCgIOzZs6eao6q4rFzN3CgOdrZwsLOKQ081jPfDlqO/7mSia3B9M0dDRGSZLL7lqCZhlxqZm5ODBC6OEihVAlIfPH66CiKi2orNF9VIlxyxS43MRCQSoaG3MxJuZSL1QQGeaOBm7pCIahVBEIyeg6866c8Cby3P7rOzszPZTSBMjqpRdp7mC+HubL3zvpD1q1fXCQm3MpHG5Iio2hUVFeHFF180dxgVEhMTY+4QymXv3r0mewYcu9WqiSAIyM7TDMZ2c2a3GplP/YePrUnNLDB6DikiopqMLUfVpECmhFKlhkgEODuy5YjMx6eOI8RiEQpkShQWKeFob/xcR0RkvEmdvCC1sdy5wLQ/nix5vjK5SsCSM+kmr5fJUTXRdqm5OEphI7bcDxrVfBJbGzT1dcXf97ORkS1jckRkJlIbkUUnR4Alx1a12K1WTbLz2aVGluPJxu4AgIxs6xhoSURUnZgcVRNty5EbB2OTBfBv7AEAyMg27qG+REQ1GZOjaqIbjO3EliMyvxYPk6MHOTKoOSibiMgAk6NqIAgCcvLZckSWo6GPCyS2YihVAnIeJu5ERKTB5Kga5MuUUKoEiEWaAdlE5mYjFsHLXfMokcxcjjsiItLH5Kga6O5Uc5JCzDvVyEJ4umkmS9PO3E5ERBpMjqpBDid/JAtUV9tylMPkiIhIH5OjapBToEmOXNmlRhZEmxxl5co4UzYRkR4mR9Ug9+EcRy5OTI7IctRxsYNIBMiVahTIlOYOh4jIYjA5qga52pYjJkdkQWxsxLqpJTjuiIjoP0yOqphcoYJMrgIAuDjyMQ1kWdxdNMlRFu9YIyLSYXJUxXILFAAAe6kNJLY2Zo6GyJCHK1uOiIgexeSoimm71DjeiCyRh4vmdv4sJkdERDpMjqqYdjA271QjS6TtVssrVECuUJk5GiIiy8DkqIrpWo6YHJEFspPYwNHeFsB/k5USEdV2TI6q2H/dahyMTZZJOzlpNp+xRkQEgMlRldI8cFYzIJstR2Sp3B6Oh8vOZ8sRERHA5KhKyeQqKFVqiAA48zZ+slBsOSIiMsTkqAppB2M7OUhgI+ahJsvk5vyw5YhjjoiIADA5qlIcjE3WwPXhLNkyuQpFct6xRkTE5KgKaSeAZJcaWTKJrRhOvGONiEiHyVEVyitkckTWQTfuiIOyiYiYHFWl/EJNt5qzA7vVKspeagNBEMwdRq3hqht3xEHZRES25g6gJsvTdqs5sOWooqQSG4hEIpy+eN/orp76Xs4IftLLxJHVTG5O2jvWrKPlKC4urkLlQ0JCqigSIqqJmBxVEblCBblSDYDJUWVk5xUZ/VBUVz7Prtys7Xb+mJgYiESiMssJggCRSISEhIRqiIqIagomR1VEO97IXmoDW1v2XpJl004EWaRQQSZXwl5q2aeG7du3V8t21qxZg9jYWOzYsaPE9bNmzcLp06fx448/6pap1WqsWrUKe/fuRU5ODjp06IA5c+agSZMm1RIzEVWeZZ8BrRi71Mia2D68Yy1fpkROvtzik6PQ0NAq38bWrVuxYsWKUrvkjh8/jr1796JBgwYGy9esWYPdu3djwYIF8PHxweLFizFixAgcPnwYUilbM4msgWWfAa1Y3sPB2E68U42shIuTFPkyJXLz5fD2cDR3OBVy4cIFnD17FgqFQjeQXxAEFBQUID4+Hl9++WW560pNTcXMmTMRHx+PZs2alVgmLS0N7733HkJDQ3Hv3j3dcrlcjs2bN2Py5MmIiIgAACxduhTh4eE4duwY+vXrV4m9JKLqwuSoiuRrb+PnnWpkJVydpEjJKEBOvnWMO9LatWsXPvzwwxLvbhSLxejatWuF6rty5Qrc3Nxw8OBBrF692iD5ATRJ17Rp0/DCCy/AyckJX3/9tW5dYmIi8vPzERYWplvm6uqKwMBAxMXFMTkishJMjqqIbo4jdquRlXB9OJO7dmZ3a7Fz50507doVS5Yswfr165Gbm4sZM2bg5MmTmDZtGvr371+h+iIjIxEZGVnq+q1btyI9PR1r167FunXrDNalpKQAAOrVq2ew3NvbG8nJyRWKQ5+2Faw0MpnM6LqpbAUFBVCr1Sapi+9V1SnP+6S9SaMsTI6qCMcckbVxeTgo29paju7evYtp06bBzc0Nbdq0wcqVK2Fvb4/evXvj5s2b2L59O5577jmTbCsxMRGrVq3Crl27Shw/VFhYCADF1tnZ2SE7O9vo7SoUisfecSeXW9d7Zm2uXbtmsvFifK+qTnnfp/KUYXJUBdRqAfkyzo5N1kU79UFeoQJqtQCxuOxfV5ZAIpHA3t4eANC0aVPcvn0bCoUCEokE7du3x+bNm02ynaKiIkyaNAlvvvkmAgICSiyjjUMul+v+rX2tg4OD0duWSCTw8/MrdT1bI6qWv7+/wftZGXyvqk553qekpKRy1cXkqAoUFCkhCIBYLIKDHQ8xWQcHO1vY2oigVAnIK1RYzTxRLVu2xE8//YROnTqhSZMmUKvVuHDhAkJCQnTdXKbw559/4q+//sKqVauwevVqAJoWHaVSiXbt2mHevHlo2rQpAM2A7caNG+tem5aWVmpCVR4ikQiOjqUPkheLOV1IVXJ0dDRZcsT3quqU530qT5cawOSoSuQVaB8bIin3G0FkbiKRCC6OUmTmFiEnv8hqkqMhQ4bgrbfeQnZ2NhYsWICePXtiypQp6N27Nw4dOoQOHTqYZDtBQUH44YcfDJbt2LEDP/zwA3bs2AFPT09IpVI4OzvjzJkzuuQoJycHV69eRXR0tEniIKKqx+SoCmjvVHPieCOyMi5O2uTIesZFPP3001i7di1u3LgBAHj//fcxceJE7N69G23atMHs2bNNsh17e/tiEzm6ubnB1tbWYHl0dDSWLFmCOnXqoEGDBli8eDF8fX3Rq1cvk8RBRFWPyVEV0CVH9kyOyLr8d8eawsyRVEz37t3RvXt3AICHh4fJxhkZY/z48VAqlZg1axZkMhlCQkKwadMmTgBJZEWYHFWBfJkSAODkwMNL1sVa71i7c+cOioqK4Ofnh+zsbCxbtgzJycno06cP/ve//xld78KFCx+7fty4cRg3bpzBMhsbG0yePBmTJ082ertEZF4cGVYFtHeqObLliKyMdpxRrhUlR6dOncKzzz6Lffv2AQDmzp2LL7/8EqmpqZg+fTr27t1r5giJyNowOaoCBRxzRFbKxfG/B9AWKVRmjqZ81qxZg65du2Ls2LHIzc3FsWPHMHLkSHz99dcYOXJktT2klohqDiZHJqZWCygoetitZs9uNbIuElsxHOxsAPx316WlS0xMxODBg+Hs7IxffvkFKpUKvXv3BgB06dIFt2/fNnOERGRtmByZWKF2jiMROMcRWSXt8wCtZVC2nZ0dlErND5JffvkFnp6eujmF/v33X7i6upozPCKyQrx6m5j+eCPOcUTWyNlRgvSsQt3zAS1dhw4dsHnzZmRnZ+Po0aMYMGAAAODy5ctYtWoV2rdvb+YIicjasOXIxAoe3qnmyC41slLOD8cdWUu32vTp05GamopJkyahYcOGePPNNwEAo0aNglwux6RJk8wcIRFZG17BTYwTQJK1c3n42c2zkm61Ro0a4dtvv0VGRgbq1q2rW7569WoEBgZyfiEiqjAmRyam7VbjBJBkrbQPS84rtI6WI0Dz6BP9xAgA2rZta55giMjqMTkyMW3LkSNbjshKabvVCotUUCrVsLW17N73Bw8eYP78+fj5559RWFgIQRAM1otEIly9etVM0RGRNTIqOYqLi0NgYCCcnJyKrcvJycEvv/yCfv36VTo4a6Qdc8Tb+Mla2UlsILUVQ65UI69QAXcXO3OH9Fhz587FyZMn0a9fP/j6+vKp50RUaUZdwV9//XXs2bMHQUFBxdZdvXoV06dPr5XJkSAIHHNENYKzoxQPcmTIK5RbfHL0yy+/YMaMGXj55ZfNHQoR1RDlTo6mTp2K5ORkAJokYO7cuXB2di5W7tatW8X6/msLuUIFlVrTpO/IOY7Iijk7SvAgR2YVcx1JpVI0atTI3GEQUQ1S7vbn3r17QxAEg/587d/a/8RiMdq2bYsFCxZUSbCWTvvAWXupDWxs2LRP1uu/O9Ysf1B2r169cPjwYXOHQUQ1SLmbNyIjIxEZGQkAiImJwdy5c9G8efMqC8wasUuNagrtoGxraDkKDAzEsmXLcOfOHQQHB8Pe3t5gvUgkwtixY80UHRFZI6P6fnbs2GHqOMp04MABrF+/Hnfu3EHjxo3x1ltv4dlnnwUAJCQkYP78+bh8+TLc3d0RExODYcOGVXuMvI2faor/bue3/OTo/fffB6C5USQuLq7YeiZHRFRRRiVHhYWFWLt2LX766ScUFhZCrVYbrBeJRDh+/LhJAgSAb775BjNmzMDUqVPRvXt3HD58GO+++y58fX3RtGlTDBkyBE8//TTmzZuHCxcuYN68eXB3d8fAgQNNFkN55Bc+nB3bgeONyLppn69WUKiASi3ARmy5j8JJTEw0dwhEVMMYdRWfP38+9u3bh9DQULRs2bJKb50VBAHLly/H4MGDMXjwYADA2LFj8ccff+Ds2bM4e/YspFIp5s6dC1tbWzRv3hy3b9/Ghg0bqj05KmDLEdUQDnY2sBGLoFILKChUwMXJOmaZzs3NRVpaGho1agQbGxvY2NiYO6RqJ6iV5g6hRuBxrN2MSo5++OEHTJgwASNHjjR1PMX8/fffuHfvHp5//nmD5Zs2bQIAjBgxAiEhIbC1/W9XwsLCsG7dOmRkZMDT07PKY9TSjTlickRWTiQSwdlRguw8OXKtIDk6c+YMlixZgsuXL0MkEmHv3r3YsGEDfH19MW3aNHOHV+X0b5TJ/+uA+QKpoR6dWJRqPqOafJRKZYlzHFWFW7duAQAKCgowbNgwdO7cGS+++CJ+/PFHAEBKSgp8fX0NXuPt7Q0AuH//frXEqKW9W43dalQTaLvWLP2OtdjYWAwbNgz29vaYNGmS7kIWGBiI7du3Y8uWLWaOkIisjVFX8a5du+LUqVMICwszdTzF5OXlAdDMs/TWW29h0qRJ+P777zFmzBhs2bIFMpms2IMl7ew0k9YVFRUZtU1BEFBQUPDYMnK5HA4ODlAqlVAolFCq1JArVJrt24qgUJQ9kFWl0pTX1GH8wFdT1FPROrRlHi1rzftU0XpKOwbmiKUq6nC013RJZefJSiynlGiSkKKiolJ/WQuCAJGoascrLVu2DD179sTy5cuhVCqxePFiAMDIkSORl5eHvXv3YsiQIVUag7npH2OnJ/8HkZg/0CpLUCt1rXBV/Rkmy2PUN6hv376YM2cOHjx4gODgYDg4OBQr87///a+ysQEAJBJNF9WwYcMQFRUFAGjZsiWuXr2KLVu2wN7eHnK54S9bbVLk6Oho1DYVCgUSEhIeW8bBwQHu7u7IzctFekYe8mWaC46NGMjKzCjXdjxdNF+43LxcpKdnGRWrqeoxto6sLMOyNWGfKlrPo8fAnLGYtA6l5nuVmZ2P9PTiyY/IUzMJ7P3791FYWFhqNY/+eDG1hIQE3d1oj17EunTpgm3btlXp9i2NSGzL5Iiokoz6Br3zzjsANLfXHzhwoNh6kUhksuRI22XWokULg+V+fn74+eef0aBBA6SlpRms0/7t4+Nj1DYlEgn8/PweW0abkLk4u0AQO0CVUQCgAM4OUnh5eZVrO64urro6vNTGj1MyRT0VrUOhUCArKwvu7u66BNZUsZiqnqqOpbRjYI5YqqIOtU0B/kpOgVwlLvEz7eKoOX3Ur1+/1AQoKSnJqBgrwsXFBenp6SWuS05OhouLS5XHQEQ1i1HJ0YkTJ0wdR6m0D7j9888/0bFjR93y69evo3Hjxmjfvj12794NlUqluzMlNjYWzZo1M3owtkgkKrPVSfsL1dbWFhKJCEVKzS9rJwdJmRdKLW28mjqMv2Caoh5j65BIDPe3JuxTRet59BiYMxZT1uHuomkRzpcpYWtrW6xVxtZWU4+dnV2JrcdA9XRH9OzZE0uXLkWLFi0QGBio225KSgrWrl2L7t27V3kMRFSzGJUcNWjQwNRxlMre3h7Dhw/H6tWr4ePjg6CgIHz77bf47bffsHXrVvj5+WHjxo2YOXMmhg8fjosXL2Lbtm2YN29etcUIAIXawdj2bM6mmsHRXgKRCFCrBRQWqSz2sz1x4kT8+eefeOmll3TPdXz33XeRkpKCevXq4d133zVzhERkbYw6261atarMMm+99ZYxVZdozJgxcHBwwNKlS5GamormzZtj5cqV6NSpEwBg48aNmD9/PqKiouDl5YUpU6boxidVl4IiTXLkwNv4qYYQi0VwtLNFvkyJfJnCYpMjNzc37N27FwcOHMDvv/+OrKwsuLi4ICYmBgMGDCi1VYuIqDQmT46cnZ3h7e1t0uQIAIYMGVLqHSdBQUHYs2ePSbdXUdoJIB3tLPMCQmQMJweJJjkqVMDL3XKTDKlUipdeegkvvfSSuUMhohrAqCt5SdP1FxQUID4+HnPnzsV7771X6cCsTQG71agGcnKQAJmFuglOLUVJN4I8jqluECGi2sFkV3JHR0eEh4dj7Nix+Pjjj/H111+bqmqrUFjE5IhqHmcHy3wA7bRp03SDvcuavdiUd88SUe1g8it5vXr1cOPGDVNXa9EUShUUSs3Ddx3sOOaIag6nh8mRpbUceXl5IT09HYGBgejXrx+6d+8Oe3t7c4dFRDWEyZIjQRCQnJyMDRs2VOvdbJZA26UmsRVDYlt1D+Elqm7a5wRaWnJ06tQpxMXF4dtvv8XGjRuxZs0a9OzZE8899xy6dOlSKx84S0SmY1RyFBAQUOr8JYIg4OOPP65UUNZGe6caB2NTTePk+DA5kimgFgSILeQxCiKRCKGhoQgNDcXs2bNx+vRpHDlyBJMmTYJYLMYzzzyD5557DqGhoeYOlYiskFFX87Fjx5aYHDk7O6N79+5o2rRpZeOyKto5jhw43ohqGAc7W4hFgFrQjKtzssCpKmxsbBAeHo7w8HAoFAqcOnUKR48exejRo+Hs7Iy+ffti2rRp5g6TiKyIUVfzcePGmToOq6a7jd8CLxxElSEWieBoL0FeoQL5BQqLTI70SSQS9OzZE76+vvD09MSuXbuwbds2JkdEVCFGN3XI5XLs378fZ86cQU5ODjw8PNCxY0dERUXBzs7OlDFaPHarUU3m5PAwOZJZ1rijR129ehVHjx7Fd999h7t376J+/foYPHgw+vbta+7QiMjKGHU1z8nJweuvv47ExETUr18fXl5euHnzJg4fPoxdu3bh888/r1UPe+SjQ6gmc3aQIBWWdzs/ACQkJOgSon/++Qc+Pj7o06cP+vbti+DgYHOHR0RWyqir+SeffIKUlBTs3LnT4GGw586dw/jx47F8+XLMmjXLZEFaugIZHx1CNZcl3s6/dOlSXULk6emJ3r1749lnnzU4HxERGcuo5OjEiRN45513ip2IOnbsiPHjx2PNmjW1Kzkq0lw0nNhyRDWQJSZH69atg42NDTp27IjQ0FCIRCL8/vvv+P3334uVFYlEGDt2rFHbWbNmDWJjY7Fjxw7dsh9//BGrV6/G33//DQ8PD/Tu3Rtvv/22bp4ltVqNVatWYe/evcjJyUGHDh0wZ84cNGnSxLidJaJqZ9TVPD8/H40aNSpxXaNGjZCVlVWZmKyKUqmGXKGdAJLJEdU8lpgcAYBKpUJcXBzi4uIeW87Y5Gjr1q1YsWIFQkJCdMvOnTuHt956C++88w569+6N27dvY/bs2cjKysKCBQsAaBKq3bt3Y8GCBfDx8cHixYsxYsQIHD58GFKptMJxEFH1M+pq/sQTT+Cnn35Cly5diq07ceJErfqFlPdwkKqtjYgTQFKNpH2ESIFMCbVagFhs/rmOSnq+o6mkpqZi5syZiI+PR7NmzQzW7d69G2FhYRg5ciQAoEmTJpgwYQJmzJiBefPmAQA2b96MyZMnIyIiAoCmCzA8PBzHjh1Dv379qixuIjIdo67mw4YNw86dOzF79mzExcXh5s2biIuLw+zZs/HFF18gOjra1HFarPxC7WBsSakTYxJZM3upDcRiEQT8N22FJRk9ejROnz5tsvquXLkCNzc3HDx4sNig7qFDh2LKlCnFXqNUKpGXl4fExETk5+cjLCxMt87V1RWBgYFltnARkeUwquWob9++uHXrFtauXYu9e/fqlkskEowdOxYvv/yyyQK0dNquBnapUU0lEongZG+L3AIF8mVKODtaVtdQXFwchgwZYrL6IiMjERkZWeK6wMBAg7/lcjm2bNmCVq1aoU6dOjh37hwAzTMm9Xl7eyM5OdnomARBQEFBQanrZTKZ0XVT2QoKCqBWq01SF9+rqlOe90kQhHI1ZBh1RS8oKMCYMWMQHR2NCxcuIDs7G8nJyXj55Zfh5uZmTJVWK183ASSTI6q5nBwkyC1QIK9QAR9zB/OILl26YO/evWjbtm21zrGmVCoxZcoUJCUlYdeuXQCAwsJCACg2tsjOzg7Z2dlGb0uhUCAhIaHU9XK53Oi6qWzXrl0z2XgxvldVp7zvU3nKVOiKnpCQgOnTp+OZZ57BmDFj4Orqim7duiE7OxudO3fGN998gxUrVqB58+YVqdaq6XerEdVUzhY6KBvQJB5Hjx7FsWPH0LBhQ3h6ehqsF4lE2LZtm0m3mZeXh3feeQdnzpzBihUrdN1v2jvW5HK57t8AUFRUBAcHB6O3J5FI4OfnV+p6tkZULX9/f4P3szL4XlWd8rxPSUlJ5aqr3MnRnTt38MYbb8DR0bHYl1QqlWLGjBnYuHEjXn31VXzzzTfw9fUtb9VWTTsgm7NjU01mqXesAUBKSgratWun+1sQBIP1j/5dWWlpaRgxYgTu3r2LDRs2GIwv0nanpaWloXHjxgavCQgIMHqbIpEIjo6Opa4Xi3kzSFVydHQ0WXLE96rqlOd9Ku/Y4HJf0devXw8PDw/s3r0b7u7uBuscHBwQHR2NZ599FoMGDcLatWsxd+7c8lZt1bQtR3zoLNVklpwc6c9BVNWys7MxePBg5OXl4fPPP4e/v7/B+oCAADg7O+PMmTO65CgnJwdXr16tVTeqUPnIVaZN3GujqjqG5b6ix8bGYvTo0cUSI32enp4YMmSIrv+9NtBeLDjmiGoybbeaJT5CROvGjRs4e/YscnNz4eHhgQ4dOuCJJ54w6TYWLFiAO3fuYOPGjahTpw7S09N16+rUqQOpVIro6GgsWbIEderUQYMGDbB48WL4+vqiV69eJo2FrJN+S+aSM+mPKUkVZcpW4nJf0dPT08s1f1GLFi2QkpJSqaCshVyhgkyuAgA42nHMEdVc2jF1hUVKqNQCbCxgriMtQRAwZ84c7N271+DkKBKJEBUVhfnz55tkmg21Wo0jR45AoVBg8ODBxdafOHECDRs2xPjx46FUKjFr1izIZDKEhIRg06ZNnACSyIqUOzmqU6cO0tLSyiz34MGDx7Yu1SQPcjQD62zEIkgl7Eemmks715FaLaBQprCo2/k3btyIffv2Yfz48ejfvz+8vLyQlpaGb775Bp999hmefPJJo2/1X7hwoe7fYrEYFy9eLPM1NjY2mDx5MiZPnmzUNqlm00/UJ3XygtTGcn5oWCO5StC1wJlyrsFyJ0chISHYv39/mTO8HjhwAC1btqx0YNbg3yzNbbsO9racAJJqNEue6+irr77C8OHD8eabb+qWNWzYEGPHjoVCocDevXtNOg8SkalIbURMjixUuZs7YmJicObMGSxcuBBFRUXF1svlcixatAi//PILXnvtNZMGaan+zda0HPFONaoNtF1rljYoOzk52eCOMX2dOnXC3bt3qzkiIrJ25b6qt2nTBtOnT8dHH32Eb775Bp07d0bDhg2hUqlw//59nDlzBpmZmXj77bcRHh5elTFbjIyHLUec44hqAyfdM9YsKzlq0KABEhMT0blz52Lrrl69ijp16pghKiKyZhVq8njttdcQEBCATZs24cSJE7oWJCcnJ3Tt2hVDhw4t9iyimuzf7P+61YhqOqeHn/N8mdLMkRh67rnnsHLlSnh7e+PZZ5+FWCzWDZ5evXo1XnnlFXOHSERWpsJX9Q4dOqBDhw4AgMzMTIjF4lr3yBCtDHarUS1iqXMdjRgxAufOncPEiRMxdepUuLu7IysrCyqVCqGhoXj77bfNHSIRWZlKXdU9PDxMFYdVStd1qzE5oppP231sad1qUqkUW7ZswcmTJ3H27Fnk5OTAzc0NISEhiIiIMHd4RGSFeFWvBI45otpEv1vN1I/kMIWIiAgmQ0RkEkyOjKRUqpGVpxlz5cBuNaoFHO0lEAFQqwXI5Cq4Oprvc//666+Xu2xVPHiWiGo2zlxopMzcIggCIBZpJsgjqunEYpHuh4C5xx0JglDu/9RqtVljJSLrwyYPI2U8nB3byUHCCSCp1nB0sEVBkRIFZr5jrTofNktEtQ9bjoz0IOfhNAYOHG9EtYeTdiJICxuUXZqCggKcOnXK3GEQkZVhy5GRtLfxO/FONapFHC3wdv579+5h9uzZiIuLg0JRclwJCQnVHBURWTNe2Y2k361GVFs4WeDt/AsWLMD58+fx0ksv4Y8//oCDgwPatm2L3377DdevX8fKlSvNHSIRWRl2qxnpQbamW82ZLUdUi+hu5y+0nFmy4+Li8M4772DWrFkYOHAgpFIpJk+ejH379iEkJAQnTpwwd4hEZGWYHBmJLUdUG+lmybaglqP8/Hy0bNkSANC8eXNdF5qNjQ1ee+01/P777+YMj4isEJMjIz3IZnJEtY92wlOFUg25QmXmaDS8vb2Rnp4OAGjSpAmys7ORlpYGAHBzc0NGRoY5wyMiK8TkyAgqlRqZeXIAHJBNtYvEVgypRHPayLOQQdkRERFYvnw5/vjjD9SrVw++vr7YvHkz8vLysG/fPvj4+Jg7RCKyMkyOjJCVVwS1WtBMisfkiGoZbetRboH5kqOXX34Ze/fuRUFBAcaPHw9XV1esWLECADBhwgRs374dISEhOHToEIYMGWK2OInIOvHKbgRXJzs08HJC84buEHMCSKplnOwlyMotMmvLkUwmw3vvvYcFCxagb9++mDFjBho2bAgA6N+/P+rXr48LFy4gKCgIoaGhZouTiKwTkyMjSGzF+HT8U3B0dMShU3+ZOxyiauXkoDlt5Jmx5eibb75BYmIiDhw4gMOHD2Pfvn1o3rw5Bg0ahP79+6Njx47o2LGj2eIjIuvGbjUj8ZEhVFtp5zrKNfOYo4CAAEybNg2nTp3C2rVr4efnh6VLlyIiIgJvv/02fvvtN7PGR0TWiy1HRFQh2js0zdlypE8sFiMiIgIRERHIy8vDt99+i2+++QbDhw9HvXr1MGDAALz11lvmDpOIrAhbjoioQhztzd+tVhpnZ2e8/PLL+Pzzz7F9+3ZIpVKsXr3a3GERkZVhyxERVYjuESJFSiiUljHXkVZqaiq+/fZbHDp0CImJiWjQoAHGjRtn7rCIyMowOSKiCrGT2sBGLIJKLeDfLBncnczbAJ2Xl4fvv/8ehw4dQlxcHGxtbfH0009jypQp6Ny5s1ljIyLrxOSIiCpEJBLB0V6C3AI50jIL4O7kXO0xKJVKnDx5EgcPHsTPP/+MoqIiBAYGYsaMGejfvz9cXFyqPSYiqjmYHBFRhTk52CK3QI70zAK0aFj9yVGXLl2Qk5MDV1dXvPjiixg0aBACAgKqPQ4iqpmYHBFRhWnHHaVlFppl+61atcKgQYPw9NNPQyqVmiUGIqq5mBwRUYU5PrydP91MydHmzZvNsl0iqh14Kz8RVZj2gcvpWQVmjoSIyPSYHBFRhfl6OsHVSYK2LbzNHQoRkckxOSKiCnOws8UrTz+JQZFPmjsUIiKTY3JEREREpIfJEREREZEeJkdEREREeqwqObp58ybatWuH/fv365YlJCQgOjoabdu2Rffu3bFp0yYzRkhERETWzmqSI4VCgUmTJqGg4L9bhzMzMzFkyBA0bdoU+/btw7hx47B8+XLs27fPjJESUU2xZs0axMTEGCwr6weZWq3GihUrEB4ejuDgYAwdOhS3b9+uzrCJqJKsJjlauXIlnJycDJZ9+eWXkEqlmDt3Lpo3b46BAwfijTfewIYNG8wUJRHVFFu3bsWKFSsMlpXnB9maNWuwe/dufPjhh9izZw9EIhFGjBgBuVxe3btAREayiuQoLi4Oe/bswaJFiwyWnzt3DiEhIbC1/W+i77CwMNy8eRMZGRnVHSYR1QCpqakYPnw4li9fjmbNmhmsK+sHmVwux+bNmzFu3DhEREQgICAAS5cuRWpqKo4dO2aO3SEiI1h8cpSTk4MpU6Zg1qxZqFevnsG6lJQU+Pr6Gizz9tZMSnf//v1qi5GIao4rV67Azc0NBw8eRHBwsMG6sn6QJSYmIj8/H2FhYbr1rq6uCAwMRFxcXLXtAxFVjsU/W23u3Llo27Ytnn/++WLrZDJZsYdO2tnZAQCKioqM3qYgCAZjm0oil8vh4OAApVIJhUJp1HZUKhUAPKxDYVQdpqqnonVoyzxa1pr3qaL1lHYMzBFLddcBAEqJAEDzXRMEocQygiBAJBIZvQ1ziIyMRGRkZInrUlJS0KJFC4Nl+j/IUlJSAKDYDzlvb28kJycbHVNZ5ySZTGZ03VS2goICqNVqk9TF96rqlOd9Ku85yaKTowMHDuDcuXM4dOhQievt7e2L9eNrkyJHR0ejt6tQKJCQkPDYMg4ODnB3d0duXi7SM/KM2o6ni+YNys3LRXp6llF1mKoeY+vIyjIsWxP2qaL1PHoMzBlLddUBACJPZwCapKCwsPQH0D76A8aalfWDTHscSiqTnZ1t9HbLOidxPFPVunbtmsk+x3yvqk5536fylLHo5Gjfvn3IyMhA9+7dDZbPmTMHmzZtQv369ZGWlmawTvu3j4+P0duVSCTw8/N7bBntB9zF2QWC2MGo7bi6uOrq8FJLjKrDVPVUtA6FQoGsrCy4u7tDIvmvvDXvU0XrKe0YmCOW6q4DAFwcNaeP+vXrl3qySUpKMrp+S1TWDzJ7e3sAmvOD9t/aMg4Oxp0ngLLPSWyNqFr+/v4G72dl8L2qOuV5n8p7TrLo5GjJkiXFPkjPPPMMxo8fj759++Lbb7/F7t27oVKpYGNjAwCIjY1Fs2bN4OnpafR2RSJRmS1P2mY5W1tbSCTGdRtoY9bUYfxFyhT1GFuHRCIxKF8T9qmi9Tx6DMwZS3XVoXm9ph47O7tSL/zW1qVWFl9f38f+IFMqlbpljRs3NigTEBBg9HbLOieJxRY/fNSq6Se+lcX3quqU530q7znJot8lHx8fNGnSxOA/APD09ESDBg0wcOBA5OXlYebMmUhKSsL+/fuxbds2jBo1ysyRE1FNFBISgvj4eN24LcDwB1lAQACcnZ1x5swZ3fqcnBxcvXoVHTt2NEfIRGQEi06OyuLp6YmNGzfi5s2biIqKwqpVqzBlyhRERUWZOzQiqoHK+kEmlUoRHR2NJUuW4MSJE0hMTMSECRPg6+uLXr16mTl6Iiovi+5WK8m1a9cM/g4KCsKePXvMFA0R1SbaH2Tz589HVFQUvLy8iv0gGz9+PJRKJWbNmgWZTIaQkBBs2rSpRg1MJ6rprC45IiKqLgsXLiy2rKwfZDY2Npg8eTImT55claERURWy6m41IiIiIlNjckRERESkh8kRERERkR4mR0RERER6mBwRERER6WFyRERERKSHyRERERGRHiZHRERERHqYHBERERHpYXJEREREpIfJEREREZEeJkdEREREepgcEREREemxNXcARERkOoJaae4QyiQIAgBAJBKZOZLSWcNxpKrD5IiIqAbJ/+uAuUMgsnrsViMiIiLSw5YjIiIrZ2dnh71795o7jHKRyWSIiYkBAOzYsQP29vZmjqhsdnZ25g6BqhmTIyIiKycSiawiyXiUvb29VcZNNR+71YiIiIj0MDkiIiIi0sPkiIiIiEgPkyMiIiIiPUyOiIiIiPQwOSIiIiLSw+SIiIiISA+TIyIiIiI9TI6IiIiI9DA5IiIiItLD5IiIiIhID5MjIiIiIj1MjoiIiIj0MDkiIiIi0sPkiIiIiEgPkyMiIiIiPUyOiIiIiPQwOSIiIiLSw+SIiKgCFAoFli5diu7du6Ndu3Z49dVX8ccff+jWJyQkIDo6Gm3btkX37t2xadMmM0ZLRMawNXcARETW5LPPPsO+ffuwcOFCNGrUCBs2bMCIESNw5MgRSKVSDBkyBE8//TTmzZuHCxcuYN68eXB3d8fAgQPNHTpZGLlKMHcIjyUImvhEIpGZIyldVR1DJkdERBVw4sQJPPfcc+jatSsAYNq0adi7dy8uXLiAW7duQSqVYu7cubC1tUXz5s1x+/ZtbNiwgckRFbPkTLq5Q6BSsFuNiKgC3N3d8dNPP+Hu3btQqVTYs2cPpFIpWrZsiXPnziEkJAS2tv/97gwLC8PNmzeRkZFhxqiJqCLYckREVAEzZ87EhAkT0LNnT9jY2EAsFmP58uVo3LgxUlJS0KJFC4Py3t7eAID79+/D09PTqG0KgoCCgoJKx24JZDKZ7t8FBQVQq9VmjKb6CYKAbdu2mTuMMhUVFWHkyJEAgPXr18POzs7MEZVNpVKV+T0RBKFc3YRMjoiIKuDGjRtwdXXF6tWr4ePjg71792Lq1KnYuXMnZDIZpFKpQXntRaWoqMjobSoUCiQkJFQqbkshl8t1/7527Vqx40WWQf990nYX1xTl2RcmR0RE5XTv3j1MnjwZW7duRceOHQEAbdq0QVJSElauXAl7e3uDiwrwX1Lk6Oho9HYlEgn8/PyMD9yC6Lcc+fv7w97e3ozRUGlq6vuUlJRUrnJMjoiIyunixYtQKBRo06aNwfLg4GCcOnUK9evXR1pamsE67d8+Pj5Gb1ckElUqubIkYvF/Q10dHR1rzEW3pqmp71N577zjgGwionKqV68eAE13kL7r16+jSZMmCAkJQXx8PFQqlW5dbGwsmjVrZvR4IyKqfkyOiIjKKSgoCB07dsTUqVPx+++/49atW1i2bBliY2MxcuRIDBw4EHl5eZg5cyaSkpKwf/9+bNu2DaNGjTJ36ERUAexWIyIqJ7FYjDVr1mDZsmWYPn06srOz0aJFC2zduhVt27YFAGzcuBHz589HVFQUvLy8MGXKFERFRZk3cCKqECZHREQV4Obmhjlz5mDOnDklrg8KCsKePXuqOSoiMiV2qxERERHpYXJEREREpIfJEREREZEeJkdEREREepgcEREREelhckRERESkh8kRERERkR4mR0RERER6mBwRERER6bH45CgrKwuzZ89Gt27d0L59e/zf//0fzp07p1ufkJCA6OhotG3bFt27d8emTZvMGC0RERFZO4tPjt599138+eef+PTTT/HVV1+hVatWGDZsGG7cuIHMzEwMGTIETZs2xb59+zBu3DgsX74c+/btM3fYREREZKUs+tlqt2/fxm+//YYvvvgC7du3BwDMnDkTp06dwuHDh2Fvbw+pVIq5c+fC1tYWzZs3x+3bt7FhwwYMHDjQzNETERGRNbLoliMPDw+sX78erVu31i0TiUQQBAHZ2dk4d+4cQkJCYGv7X44XFhaGmzdvIiMjwxwhExERkZWz6JYjV1dXREREGCw7evQo/vnnH3Tt2hVLly5FixYtDNZ7e3sDAO7fvw9PT0+jtisIAgoKCh5bRi6Xw8HBAUqlEgqF0qjtqFQqAHhYh8KoOkxVT0Xr0JZ5tKw171NF6yntGJgjluquAwCUEgEAUFRUBEEQSiwjCAJEIpHR2yAiMgeLTo4eFR8fjxkzZqBnz56IjIzEggULIJVKDcrY2dkB0JywjaVQKJCQkPDYMg4ODnB3d0duXi7SM/KM2o6ni+aikZuXi/T0LKPqMFU9xtaRlWVYtibsU0XrefQYmDOW6qoDAESezgA0P0QKCwtLLffod5SIyNJZTXJ0/PhxTJo0CcHBwfj0008BAPb29pDL5QbltEmRo6Oj0duSSCTw8/N7bBntdl2cXSCIHYzajquLq64OL7XEqDpMVU9F61AoFMjKyoK7uzskkv/KW/M+VbSe0o6BOWKp7joAwMVRc/qoX79+qQlQUlKS0fUTEZmLVSRHO3fuxPz589GrVy8sWbJEdyL29fVFWlqaQVnt3z4+PkZvTyQSlZlcabsKbG1tIZEY121gY2OjV4fxFylT1GNsHRKJxKB8Tdinitbz6DEwZyzVVYfm9Zp67Ozs4OBQ8g8EdqkRkTWy6AHZAPD555/jgw8+wGuvvYZly5YZ/EINCQlBfHy8bgwFAMTGxqJZs2ZGjzciIiKi2s2ik6ObN2/io48+Qq9evTBq1ChkZGQgPT0d6enpyM3NxcCBA5GXl4eZM2ciKSkJ+/fvx7Zt2zBq1Chzh05ERERWyqK71b7//nsoFAocO3YMx44dM1gXFRWFhQsXYuPGjZg/fz6ioqLg5eWFKVOmICoqykwRExERkbWz6ORo9OjRGD169GPLBAUFYc+ePdUUEREREdV0Ft2tRkRERFTdmBwRERER6WFyRERERKSHyRERERGRHiZHRERERHqYHBERERHpYXJEREREpIfJEREREZEeJkdEREREepgcEREREelhckRERESkh8kRERERkR4mR0RERER6mBwRERER6WFyRERERKSHyRERUQUdOHAAffv2RZs2bdCvXz8cPXpUty4hIQHR0dFo27Ytunfvjk2bNpkxUiIyBpMjIqIK+OabbzBjxgy8/PLLOHz4MPr27Yt3330X58+fR2ZmJoYMGYKmTZti3759GDduHJYvX459+/aZO2wiqgBbcwdARGQtBEHA8uXLMXjwYAwePBgAMHbsWPzxxx84e/Yszp49C6lUirlz58LW1hbNmzfH7du3sWHDBgwcONDM0RNRebHliIionP7++2/cu3cPzz//vMHyTZs2YdSoUTh37hxCQkJga/vf786wsDDcvHkTGRkZ1R0uERmJLUdEROV069YtAEBBQQGGDRuGq1evomHDhnjzzTcRGRmJlJQUtGjRwuA13t7eAID79+/D09PTqO0KgoCCgoJKxW4pZDKZ7t8FBQVQq9VmjIZKU1PfJ0EQIBKJyizH5IiIqJzy8vIAAFOnTsVbb72FSZMm4fvvv8eYMWOwZcsWyGQySKVSg9fY2dkBAIqKiozerkKhQEJCgvGBWxC5XK7797Vr14odL7IMNfl9Ks++MDkiIioniUQCABg2bBiioqIAAC1btsTVq1exZcsW2NvbG1xUgP+SIkdHx0pt18/Pz+jXWxL9Fgl/f3/Y29ubMRoqTU19n5KSkspVjskREVE5+fr6AkCxrjM/Pz/8/PPPaNCgAdLS0gzWaf/28fExersikahSyZUlEYv/G+rq6OhYYy66NU1NfZ/K06UGcEA2EVG5BQYGwsnJCX/++afB8uvXr6Nx48YICQlBfHw8VCqVbl1sbCyaNWtm9HgjIqp+TI6IiMrJ3t4ew4cPx+rVq3H48GH8888/+Oyzz/Dbb79hyJAhGDhwIPLy8jBz5kwkJSVh//792LZtG0aNGmXu0ImoAtitRkRUAWPGjIGDgwOWLl2K1NRUNG/eHCtXrkSnTp0AABs3bsT8+fMRFRUFLy8vTJkyRTc+iYisA5MjIqIKGjJkCIYMGVLiuqCgIOzZs6eaIyIiU2K3GhEREZEeJkdEREREepgcEREREelhckRERESkh8kRERERkR4mR0RERER6mBwRERER6WFyRERERKSHyRERERGRHiZHRERERHqYHBERERHpYXJEREREpIfJEREREZEeJkdEREREepgcEREREemxNXcARERkuQRBQFFRkcnqk8lkJf7bFOzs7CASiUxap7Xg+2RaTI6IiKhEgiBg6tSpSEhIqJL6Y2JiTFpfy5YtsWjRIou/8Joa3yfTY7caERERkR62HBERUYlEIhEWLVpk0u4aQNPSoa3flKyhu6Yq8H0yPSZHRERUKpFIBHt7e3OHQWXg+2Ra7FYjIiIi0sPkiIiIiEgPkyMiIiIiPUyOiIiIiPQwOSIiIiLSw+SIiIiISA+TIyIiIiI9TI6IiIiI9DA5IiIiItLD5IiIiIhIT41IjtRqNVasWIHw8HAEBwdj6NChuH37trnDIiIiIitUI5KjNWvWYPfu3fjwww+xZ88eiEQijBgxAnK53NyhERERkZWx+uRILpdj8+bNGDduHCIiIhAQEIClS5ciNTUVx44dM3d4REREZGWsPjlKTExEfn4+wsLCdMtcXV0RGBiIuLg4M0ZGRERE1kgkCIJg7iAq44cffsC4cePw559/wt7eXrf87bffhkwmw7p16ypU3x9//AFBECCRSB5bThAEiMViyIqUUBt5CG1txJBKbCCTK6FWG/82mKIeY+pQq9UQiw3za2vfp4rWU9IxMFcs1VkHAIhFItjb2UKtVkMkEpVYRqFQQCQSoX379kZvp7bTnpOkUqm5QyGyenK5vFznJNtqiqfKFBYWAkCxE4ednR2ys7MrXJ/2JF/ayf7RcvZ2lT+E9lLTvA2mqMeSYjFVPYyl6uoA8NjkUCQSlfldosfj8SMynfKek6w+OdK2FsnlcoOWo6KiIjg4OFS4vnbt2pksNiKiyuI5iaj6Wf2Yo3r16gEA0tLSDJanpaXB19fXHCERERGRFbP65CggIADOzs44c+aMbllOTg6uXr2Kjh07mjEyIiIiskZW360mlUoRHR2NJUuWoE6dOmjQoAEWL14MX19f9OrVy9zhERERkZWx+uQIAMaPHw+lUolZs2ZBJpMhJCQEmzZt4t0dREREVGFWfys/ERERkSlZ/ZgjIiIiIlNickRERESkh8kRERERkR4mR0RERER6mBwRERER6WFyRERERKSHyVEZ7t27B39//2L/7d27FwCQkJCA6OhotG3bFt27d8emTZvMHLHprFmzBjExMQbLytpftVqNFStWIDw8HMHBwRg6dChu375dnWGbTEn7P3369GKfhW7duunW14T9z8rKwuzZs9GtWze0b98e//d//4dz587p1temzwCZFj8b1qek82CtINBjnThxQmjTpo2QmpoqpKWl6f4rLCwUHjx4IHTq1EmYOXOmkJSUJHz11VdCmzZthK+++srcYVfali1bBH9/fyE6Olq3rDz7u3LlSqFz587Czz//LCQkJAhDhw4VevXqJRQVFZljN4xW0v4LgiBERUUJn376qcFnISMjQ7e+Juz/kCFDhP79+wtxcXHCjRs3hA8++EAICgoSkpKSatVngEyPnw3rUtp5sDZgclSGzz77TOjfv3+J69auXSuEh4cLCoVCt+yTTz4RevfuXV3hmVxKSoowbNgwoW3btkKfPn0MvhRl7W9RUZHQrl074fPPP9etz87OFoKCgoTDhw9X305UwuP2X6lUCm3atBGOHTtW4mtrwv7funVLaNGihRAfH69bplarhV69egnLli2rFZ8Bqhr8bFiPx50Hawt2q5Xh2rVr8PPzK3HduXPnEBISAlvb/57CEhYWhps3byIjI6O6QjSpK1euwM3NDQcPHkRwcLDBurL2NzExEfn5+QgLC9Otd3V1RWBgIOLi4qptHyrjcft/69YtFBUVoXnz5iW+tibsv4eHB9avX4/WrVvrlolEIgiCgOzs7FrxGaCqwc+G9XjcebC2qBHPVqtK169fh5eXF1599VXcunULTZo0wZgxYxAeHo6UlBS0aNHCoLy3tzcA4P79+/D09DRHyJUSGRmJyMjIEteVtb8p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6OohIsvi7u4OJycn3L9/v9QyBQUFkMvluoTo0R9u2gRLf+xkeZV2rtM/H2nr53nEPJgckQFXV1f06tULhw4dwvDhw3HgwAFERkbC09MT586dw9SpUxEdHY1hw4bpftl8/PHHiI+PL7E+Nzc3ZGZmQqlUGiRI2tvvPTw8yh1b3759sW3bNly6dAlHjhzBM888A4lEUqyci4sLmjZtiiVLlpRYT0Wa0n18fDB37lzMmTMHiYmJ+O6777Bhwwa4ublh3rx5mDRpEm7cuIEtW7agffv2kEqlKCwsxN69e8u9DSKqfl27dsWZM2dQVFQEOzu7Yuv379+P+fPn4/PPPy9XfSKRCICmNcrGxgYAkJ+fb7qAqVqxW42KGTRoEBITE3H27FmcP39e1zd+/vx5qNVqjB8/XpcYqVQqnD59GkDJv6BCQ0OhUqlw5MgRg+XaKQO0rSzl0bZtWzRo0ACHDh3Cjz/+WOq0AqGhoUhOToanpyfatGmj+y82NhYbN27UnbjKcv78eTz11FO4ePEiRCIRWrZsiQkTJqBFixa6bsf4+Hj07t0bYWFhkEqlAKCbK8qYX5REVD2GDh2KrKwsLF26tNi6jIwMbNy4EU2aNCnW9V4a7cDt5ORk3TLelGG92HJExYSFhaFhw4Z477334Ovri65duwLQDEYGgPfffx8DBw5ETk4Odu7cicTERACaZmjtCUKrW7du6NSpE+bMmYO0tDQEBgbi7Nmz2LBhA6KionSDscurT58+2L59O9zd3REaGlpimQEDBmDnzp0YMmQIRo8ejXr16uH06dPYsGEDoqOjS2xtKklgYCDs7e0xZcoUjBs3DnXr1sXp06eRkJCgu5suKCgIhw4dQqtWreDr64vz589j3bp1EIlEurFLRGR52rZti7fffhvLli3DjRs3EBUVBQ8PD/z111/YvHkz8vPzsX79el2LUFkiIiKwYMECvPfeexgxYgRSUlKwatUqODk5VfGeUFVgyxEVIxKJMGDAANy6dQsDBgzQ9X136tQJs2fPxvnz5zFixAgsWLAA9evXx6pVqwCgxK41kUiEdevW4ZVXXsH27dsxcuRIfPfdd5gwYQLmz59f4dj69u0LhUKBZ599VhfXoxwdHbFr1y506NABixcvxogRI/DDDz9g4sSJFZrUzc7ODps3b8aTTz6J+fPnY9iwYThx4gTef/99DBgwAACwcOFCBAcH44MPPsDYsWNx/PhxzJs3D127dsW5c+cqvH9EVH3efPNNXQK0YMECjBw5Ejt27EC3bt3wzTfflDpwuiTNmjXDokWLcP/+fYwcORLbtm3DBx98AG9v7yrcA6oqIoGjvYiIiIh02HJEREREpIfJEREREZEeJkdEREREepgcEREREelhckRERESkh8kRERERkR4mR0RERER6mBwRERER6WFyRERERKSHyRERERGRHiZHRERERHqYHBERERHp+X9iFHMRFespVAAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 600x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkcAAAHkCAYAAAA0I4sqAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABff0lEQVR4nO3de1xUdf7H8ddwGRgERFFERYUgJSxvgWJ5b+26tpm2rRvmvbtuVlitllqa+ZOi1KxULLu4uqXdzN3NrOxmqGSpAZYl5oWLIoJch8v8/mCZHQQVhssAvp+PhwXnfOfM5zBzzrzne77nHIPFYrEgIiIiIgA4OboAERERkaZE4UhERETEhsKRiIiIiA2FIxEREREbCkciIiIiNhSORERERGwoHImIiIjYUDgSERERsaFwdBHR9T5FREQuTOGoiRg/fjw9evSw/gsNDaVv377ceuutvPnmm5SWllZqP2LECB577LEaL3/btm08+uijF2z32GOPMWLECLuf51zi4+Pp0aMH8fHxNX7M7t27GTp0KKGhoYSFhREWFsbEiRPrXMu5nL3udbVv3z6io6MZNmwYvXr14pprrmHOnDkcOXKkUrvx48czfvz4entekaZI28O5FRUV8frrrzNmzBjCw8OJiIjg9ttv57333qOsrMzazp79qNjHxdEFyP+EhYUxd+5cAEpLS8nOzmb79u0888wzJCQkEBsbi8FgAGD58uV4enrWeNmvv/56jdrdd9993HnnnbWu/UJ69uzJhg0bCAkJqfFjAgMDefXVVzGbzbi6uuLu7k63bt3qvbaG8Pbbb/PMM88wYMAAHn74Yfz8/Pj9999ZvXo1n3zyCa+99ho9e/Z0dJkijULbw7mdPHmSqVOnkpqayvjx4+nVqxdlZWV88cUX/P3vf2fnzp0888wz1n2/NA6FoybE09OTPn36VJo2YsQIgoKCWLRoESNGjODmm28GyoNUQ+jatWuDLLe6dbuQdu3a0a5duwappyElJCSwcOFC7rjjDmbPnm2dPmDAAK655hpuvfVWHn/8cT788EMHVinSOLQ9nN+jjz5KWloaGzZsIDAw0Dp92LBhBAQEsGTJEoYPH861117ruCIvQjqs1gyMHz8ePz8/1q9fb5129uGuLVu2cPPNN9OrVy8iIyN55JFHyMjIsD5+586d7Ny509olW9E9u379eoYPH85VV13F119/Xe2hpeLiYhYsWEBERAQRERE8+uijnDp1yjq/usccPXqUHj16sGnTJqD67uC9e/cyZcoUrrzySiIjI3nooYdIT0+3zk9OTuaBBx4gMjKSnj17MnjwYBYsWEBhYaG1TVFRES+99BLXX389V1xxBddeey0rV66s1BVdnezsbB5//HEGDBhAREQES5YsqfYxn376KbfeeitXXHEFV199NQsWLCA/P/+8y46Li8PLy4uHHnqoyry2bdvy2GOPce2115Kbm2udbrFYWLVqlfWQw+23386+ffus85ctW0aPHj2qLK9Hjx4sW7YM+N/f/LXXXuOGG26gf//+bNq0iWXLljFy5Ei++OILRo0axeWXX851113He++9d971EKkPLWF7+Oijj+jRowfJycmVpm/fvp0ePXqwd+9eAN58803rvmjw4MHMmzev0nqdLSkpia+//popU6ZUCkYV7rzzTu644w5atWpVafpvv/3GlClT6N27N1dffTUxMTGUlJRU+3c419/sscceY8KECcydO5fw8HBGjx5NSUkJPXr04O2332b27Nn079+fvn37MmPGDE6ePHnO9WiJ1HPUDDg7OzNw4EC2bNlCSUkJLi6VX7aEhAQeeeQR7rvvPiIiIkhLS2PJkiU8/PDDvPnmm8ydO5fo6GgA5s6dS0hICD/99BMAsbGxzJ8/n6KiIvr06cPmzZurPP+//vUvevXqxbPPPsupU6eIiYnh8OHDlcJabSUnJ3PHHXfQq1cvFi9eTGlpKc899xxTpkzh/fff59SpU9xxxx306dOHZ599FqPRyBdffMHatWtp164d99xzDxaLhXvuuYcffviB+++/n8suu4z4+HheeOEFjhw5wtNPP13tc5eVlTF16lSOHj3KI488gq+vL6tXr2bv3r34+flZ23300Uc88sgjjBo1igcffJBjx44RGxvLwYMHee2116rt5rZYLHz99deMGDECk8lU7fNff/31VaYlJCRgNpt54oknMJvNLF68mHvuuYft27dXeb0vJDY2lieffBJvb28uv/xyNm7cyIkTJ3jqqae499576dy5M3FxcTz22GP06tWL4ODgWi1fpKZayvYwcuRIWrVqxccff0xoaKh1+ubNmwkKCqJXr158/PHHLF68mEcffZQePXrw22+/sXjxYgoLC3n22Werre2rr74COOdYR6PRyJNP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m7z99tuZNm0aEydOpLCwkBdeeIHLL7+cQYMG8dNPP7FixQpWrlxJnz59OHz4sPWCkBXHvocMGcKAAQOYO3cuGRkZhIWFsXPnTlatWsXo0aPPecHJgQMHMmjQIObMmUNmZiadO3fmjTfe4NSpU9beMGdnZ2bOnMmTTz6Js7Mzw4cPJycnhxUrVpCenn7eC9b16dOHv/3tb7zwwgv8+uuvjB49mjZt2lj/Lnl5eaxcubJWF3UbOnQoixYt4oknnmDatGmkpaWxfPnyKqf4ijQ1LWl7CAkJoWfPnqxbt46RI0dav7hB+WG3uXPnsnjxYoYMGUJOTg7Lly8nMDDwvIPI58+fz4QJE7jtttu488476dOnD3l5efznP/9h8+bN3HbbbdZxTTU1bNgwPv74Y3r16kVQUBDvvfcehw8ftnu9L0YKR01IYmIit99+O1De0+Pr60tQUBDPPvvseTeOIUOGEBMTw5o1a6yDsK+88kreeOMN6xilO+64g/379zNt2jQWLVpU6ZT1C/nzn/9MYWEh999/P0ajkVGjRhEdHW3dmV199dU8+uijvPnmm3zyySf07NmT5cuX85e//OWcywwLC+PNN9/kueee45577sFoNPLHP/6RRx55BKPRyN13301WVhZvvPEGL730Eh07duRPf/oTBoOBV199lezsbFq3bs2rr77K0qVLreEmICCAmTNnMmnSpPOu0/Lly4mJiWHp0qUUFRVx44038uc//5lt27ZZ29x22220atWK1atXs2HDBjw8POjXrx8xMTF06dLlvMu/9957CQsL4+2332bRokWcPn0af39/hgwZwj333EOnTp1q/PcHCAoKYvHixbz88svcddddBAcH8/TTT5/zcgUiTUlL2h7+9Kc/8dNPP1lP/Kjwl7/8heLiYtavX8+6detwd3dn4MCBREdH4+rqes7lderUiQ0bNrB27Vo+/vhjVq1ahaurK5dccglLlizhpptuqnWNjz/+OCUlJSxZsgQXFxduvPFGHn74YebMmVPrZV2sDJaLbZSVNCm//PILY8eOZdq0adx7773n7W0SERFpDBpzJA5jNpvJy8tj1qxZLFu2jISEBEeXJCIiosNq4jipqalMmjQJJycnRo8eXet7r4mIiDQEHVYTERERsaHDaiIiIiI2FI5EREREbCgciYiIiNjQgOyz7NmzB4vFct7rUoiI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7Zk1a5Z1fBLAjBkzKCkpYc6cORQWFhIREUFcXJzGNYiIiIhdHB6OACZNmsSkSZOqnderVy82bNhwzsc6OzsTHR1NdHR0Q5UnIiIiFxGHXudIREREpKlROBIRERGxoXAkIiIiYkPhSERERMSGwpGIiIiIDYUjERERERsKRyIiIiI2FI5EREREbCgciYiIiNhQOBIRERGxoXAkIiIiYkPhSERERMSGwpGIiIiIDYUjERERERsKRyIiIiI2FI5EREREbCgciYiIiNhQOBIRERGxoXAkIiIiYkPhSERERMSGwpGIiIiIDYUjERERERsKRyIiIiI2FI5EREREbCgciYiIiNhQOBIRERGxoXAkIiIiYkPhSERERMSGwpGIiIiIDYUjERERERsKRyIiIiI2FI5EREREbCgciYiIiNhQOBIRERGxoXAkIiIiYkPhSERERMSGwpGIiIiIDYUjERERERsKRyIiIiI2FI5EREREbCgciYiIiNhQOBIRERGxoXAkIiIiYkPhSERERMSGwpGIiIiIDYUjERERERsKRyIiIiI2FI5EREREbCgciYiIiNhQOBIRERGx4fBwdOzYMXr06FHl3zvvvANAUlISUVFR9OnTh2HDhhEXF1fp8WVlZSxdupTBgwfTu3dvJk+ezOHDhx2xKiIiItICuDi6gAMHDuDm5sann36KwWCwTvfy8iIrK4tJkybxhz/8gfnz5/PDDz8wf/58fHx8GDNmDAArVqxg/fr1LFq0iA4dOrBkyRKmTZvG5s2bMRqNjlotERERaaYcHo5+/vlngoKC8PPzqzJv7dq1GI1G5s2bh4uLC8HBwRw+fJhVq1YxZswYzGYza9asITo6mqFDhwIQGxvL4MGD2bp1KzfddFNjr46IiIg0cw4/rHbgwAFCQkKqnbd7924iIiJwcflfhouMjOTQoUNkZmaSnJxMXl4ekZGR1vne3t6EhYWxa9euBq9dREREWh6Hh6Off/6ZzMxM/vrXv3LVVVcxbtw4vvrqKwDS0tLw9/ev1L6ih+n48eOkpaUB0LFjxyptUlNTG6F6EWmJajuW8cSJEzz00EMMGDCAAQMG8Le//c26fxKR5sehh9XMZjMpKSmYTCZmzZqFh4cHH374IdOmTeO1116jsLCwyrghNzc3AIqKiigoKACotk12drbddVksFvLz8y9Yu8lkoqSkhOLiErufq6TEGYCCggIsFovdy7lYVbwHKv4vjlHT18FisVQaW9hU1XYs48yZMyktLeW1114DYP78+dx3331s2rSpsUsXkXrg0HBkNBrZtWsXLi4u1h3O5Zdfzq+//kpcXBzu7u6YzeZKjykqKgLAw8MDd3d3oDyoVPxc0cZkMtldV3FxMUlJSedtYzKZ8PHx4UzuGU5k5tr9XIYyTwAOHTqkD/g6SElJcXQJQs1eh6Z+okRtxzLm5OSwa9cuXn75ZcLCwgC46667uO+++8jKyqJNmzaNvg4iUjcOH5Dt4eFRZVr37t35+uuv8ff3JyMjo9K8it87dOhASUmJdVrXrl0rtQkNDbW7JldX13OOg6pQEdq8PL2wONkfxNp4l4e6oKAg9RzZoaCggJSUFAIDA+sUiKVuavo6HDx4sBGrss+FxjKeHY7c3Nzw8PDg/fffp3///gB88MEHBAYG0rp160atXUTqh0PDUXJyMuPGjWPVqlWEh4dbp+/fv5+QkBAuu+wy1q9fT2lpKc7O5YefduzYQVBQEL6+vnh5eeHp6Ul8fLw1HOXk5JCYmEhUVJTddRkMhmpD29ltAFxcXHB1tf8wQcVgc32w143JZLrgayYN70KvQ3M4pFbbsYxubm4sXLiQp556ivDwcAwGA+3bt+ett97Cycn+YZ01ObwvIrVT00P7Dg1H3bt359JLL2X+/PnMnTuXNm3a8M9//pMffviBd999l3bt2rF69Wpmz57N1KlT2bt3L2vXrmX+/PlAefd8VFQUMTExtG3bls6dO7NkyRL8/f0ZOXKkI1dNRJqp2o5ltFgsHDhwgL59+zJ16lRKS0uJjY3l/vvv5x//+Aeenp521VGTw/siUns1ObTv0HDk5OTEK6+8QkxMDA8++CA5OTmEhYXx2muv0aNHDwBWr17NwoULGT16NO3bt2fWrFmMHj3auowZM2ZQUlLCnDlzKCwsJCIigri4uCY/rkFEmqbajmX8+OOPWbduHZ9//rk1CL3yyisMHz6cjRs3MmHCBLvqqMnhfRGpnZoe2nf4mKO2bdvyzDPPnHN+r1692LBhwznnOzs7Ex0dTXR0dEOUJyIXmYrDaTUdy5iQkEBQUFClHqLWrVsTFBRUpxMFanJ4X0Rqp6aH9h1+nSMRkaYkNDTUOpaxQsVYRtuxkRU6duzI4cOHrWfSQvmhuaNHj9KtW7dGqVlE6pfCkYiIDduxjNu2bSM5OZmZM2daxzKWlpZy4sQJCgsLAbjlllsAePDBB0lOTra2NxqN3HrrrQ5cExGxl8KRiMhZZsyYwdixY5kzZw7jxo3D2dnZOpYxNTWVQYMGsWXLFqD8LLZ169ZhsViYMGECkyZNwtXVlX/84x94e3s7eE1ExB4OH3MkItLUnG8sY0BAAAcOHKg0LTg4mFdeeaWxyhORBqaeIxEREREbCkciIiIiNhSORERERGwoHImIiIjYUDgSERERsaFwJCIiImJD4UhERETEhsKRiIiIiA2FIxEREREbCkciIiIiNhSORERERGwoHImIiIjYUDgSERERsaFwJCIiImJD4UhERETEhsKRiIiIiA2FIxEREREbCkciIiIiNhSORERERGwoHImIiIjYUDgSERERsaFwJCIiImJD4UhERETEhsKRiIiIiA2FIxEREREbCkciIiIiNhSORERERGwoHImIiIjYUDgSERERsaFwJCIiImJD4UhERETEhsKRiIiIiA2FIxEREREbCkciIiIiNhSORERERGwoHImIiIjYUDgSERERsaFwJCIiImJD4UhERETEhsKRiIiIiA2FIxEREREbCkciIiIiNhSORERERGwoHImIiIjYUDgSERERsaFwJCIiImKjSYWjQ4cO0bdvXzZt2mSdlpSURFRUFH369GHYsGHExcVVekxZWRlLly5l8ODB9O7dm8mTJ3P48OHGLl1ERERaiCYTjoqLi3nkkUfIz8+3TsvKymLSpEkEBgayceNGpk+fzosvvsjGjRutbVasWMH69etZsGABGzZswGAwMG3aNMxmsyNWQ0RERJq5JhOOli1bRqtWrSpN++c//4nRaGTevHkEBwczZswYJk6cyKpVqwAwm82sWbOG6dOnM3ToUEJDQ4mNjSU9PZ2tW7c6YjVERESkmWsS4WjXrl1s2LCBxYsXV5q+e/duIiIicHFxsU6LjIzk0KFDZGZmkpycTF5eHpGRkdb53t7ehIWFsWvXrkarX0RERFoOh4ejnJwcZs2axZw5c+jYsWOleWlpafj7+1ea5ufnB8Dx48dJS0sDqPI4Pz8/UlNTG7BqERERaalcLtykYc2bN48+ffowatSoKvMKCwsxGo2Vprm5uQFQVFREQUEBQLVtsrOz7a7JYrFUGvtUHbPZjMlkoqSkhOLiErufq6TEGYCC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",
      "text/plain": [
       "<Figure size 600x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Realizando a EDA\n",
    "eda(df_dsa)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Automação do Processo de Divisão em Dados de Treino e Teste"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1000 entries, 0 to 999\n",
      "Data columns (total 7 columns):\n",
      " #   Column             Non-Null Count  Dtype  \n",
      "---  ------             --------------  -----  \n",
      " 0   Idade              1000 non-null   int64  \n",
      " 1   UsoMensal          1000 non-null   int64  \n",
      " 2   Plano              1000 non-null   object \n",
      " 3   SatisfacaoCliente  1000 non-null   int64  \n",
      " 4   TempoContrato      1000 non-null   object \n",
      " 5   ValorMensal        1000 non-null   float64\n",
      " 6   Churn              1000 non-null   int64  \n",
      "dtypes: float64(1), int64(4), object(2)\n",
      "memory usage: 54.8+ KB\n"
     ]
    }
   ],
   "source": [
    "df_dsa.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Função para automatizar a divisão em treino e teste\n",
    "def split_dataset(data, target_column, test_size, random_state = 42):\n",
    "    \"\"\"\n",
    "    Divide o dataset em conjuntos de treino e teste.\n",
    "\n",
    "    Parâmetros:\n",
    "    - data (DataFrame): O DataFrame completo.\n",
    "    - target_column (str): O nome da coluna alvo (target).\n",
    "    - test_size (float): A proporção do conjunto de teste.\n",
    "    - random_state (int): Seed para a geração de números aleatórios (padrão é 42).\n",
    "\n",
    "    Retorna:\n",
    "    - X_train (DataFrame): Conjunto de treino para as variáveis independentes.\n",
    "    - X_test (DataFrame): Conjunto de teste para as variáveis independentes.\n",
    "    - y_train (Series): Conjunto de treino para a variável alvo.\n",
    "    - y_test (Series): Conjunto de teste para a variável alvo.\n",
    "    \"\"\"\n",
    "\n",
    "    # Dados de entrada\n",
    "    X = data.drop(target_column, axis = 1)\n",
    "    \n",
    "    # Dados de saída\n",
    "    y = data[target_column]\n",
    "    \n",
    "    # Divisão em treino e teste\n",
    "    X_train, X_test, y_train, y_test = train_test_split(X, \n",
    "                                                        y, \n",
    "                                                        test_size = test_size, \n",
    "                                                        random_state = random_state)\n",
    "\n",
    "    return X_train, X_test, y_train, y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Uso da função anterior\n",
    "valor_test_size = 0.3\n",
    "X_train, X_test, y_train, y_test = split_dataset(df_dsa, 'Churn', test_size = valor_test_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((700, 6), (300, 6), (700,), (300,))"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Verificando o tamanho dos conjuntos de treino e teste\n",
    "X_train.shape, X_test.shape, y_train.shape, y_test.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Pré-Processamento dos Dados\n",
    "\n",
    "O pré-processamento, especialmente a aplicação de técnicas de encoding e a normalização de dados, deve idealmente ser feito após a divisão do dataset em conjuntos de treino e teste. Isso evita o vazamento de informações do conjunto de teste para o conjunto de treino, o que pode acontecer se o pré-processamento for feito antes da divisão. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Idade</th>\n",
       "      <th>UsoMensal</th>\n",
       "      <th>Plano</th>\n",
       "      <th>SatisfacaoCliente</th>\n",
       "      <th>TempoContrato</th>\n",
       "      <th>ValorMensal</th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>56</td>\n",
       "      <td>52</td>\n",
       "      <td>Premium</td>\n",
       "      <td>1</td>\n",
       "      <td>Curto</td>\n",
       "      <td>75.48</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>69</td>\n",
       "      <td>65</td>\n",
       "      <td>Basico</td>\n",
       "      <td>4</td>\n",
       "      <td>Curto</td>\n",
       "      <td>79.25</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>46</td>\n",
       "      <td>76</td>\n",
       "      <td>Standard</td>\n",
       "      <td>3</td>\n",
       "      <td>Longo</td>\n",
       "      <td>183.56</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>32</td>\n",
       "      <td>42</td>\n",
       "      <td>Basico</td>\n",
       "      <td>2</td>\n",
       "      <td>Longo</td>\n",
       "      <td>162.50</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>60</td>\n",
       "      <td>74</td>\n",
       "      <td>Standard</td>\n",
       "      <td>2</td>\n",
       "      <td>Longo</td>\n",
       "      <td>186.23</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Idade  UsoMensal     Plano  SatisfacaoCliente TempoContrato  ValorMensal  \\\n",
       "0     56         52   Premium                  1         Curto        75.48   \n",
       "1     69         65    Basico                  4         Curto        79.25   \n",
       "2     46         76  Standard                  3         Longo       183.56   \n",
       "3     32         42    Basico                  2         Longo       162.50   \n",
       "4     60         74  Standard                  2         Longo       186.23   \n",
       "\n",
       "   Churn  \n",
       "0      0  \n",
       "1      0  \n",
       "2      0  \n",
       "3      0  \n",
       "4      1  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Amostra dos dados\n",
    "df_dsa.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Selecionando as variáveis categóricas\n",
    "categorical_cols = df_dsa.select_dtypes(include = ['object']).columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Plano', 'TempoContrato'], dtype='object')"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "categorical_cols"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### One-Hot Encoding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Aplicando One-Hot Encoding separadamente aos conjuntos de treino e teste\n",
    "encoder = OneHotEncoder(sparse_output = False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Por padrão, o OneHotEncoder retorna uma matriz esparsa quando você o utiliza para transformar dados. Uma matriz esparsa é uma maneira eficiente de armazenar dados com muitos zeros (valores não presentes). No entanto, se você definir sparse_output=False, o encoder retornará uma matriz densa (numpy array) em vez de uma matriz esparsa. Uma matriz densa é mais fácil de trabalhar e entender, mas pode consumir mais memória se os dados forem grandes e a maioria dos valores for zero."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Treinando o encoder com o conjunto de treino e transformando ambos treino e teste\n",
    "X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_cols]))\n",
    "X_test_encoded = pd.DataFrame(encoder.transform(X_test[categorical_cols]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Renomeando as colunas para corresponderem aos nomes das categorias\n",
    "X_train_encoded.columns = encoder.get_feature_names_out(categorical_cols)\n",
    "X_test_encoded.columns = encoder.get_feature_names_out(categorical_cols)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Removendo as colunas categóricas originais e adicionando as codificadas\n",
    "X_train_preprocessed = X_train.drop(categorical_cols, axis = 1).reset_index(drop = True)\n",
    "X_train_preprocessed = pd.concat([X_train_preprocessed, X_train_encoded], axis = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Idade</th>\n",
       "      <th>UsoMensal</th>\n",
       "      <th>SatisfacaoCliente</th>\n",
       "      <th>ValorMensal</th>\n",
       "      <th>Plano_Basico</th>\n",
       "      <th>Plano_Premium</th>\n",
       "      <th>Plano_Standard</th>\n",
       "      <th>TempoContrato_Curto</th>\n",
       "      <th>TempoContrato_Longo</th>\n",
       "      <th>TempoContrato_Medio</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>65</td>\n",
       "      <td>80</td>\n",
       "      <td>4</td>\n",
       "      <td>174.10</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>49</td>\n",
       "      <td>18</td>\n",
       "      <td>3</td>\n",
       "      <td>101.59</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>19</td>\n",
       "      <td>91</td>\n",
       "      <td>4</td>\n",
       "      <td>87.93</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>52</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>90.74</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>62</td>\n",
       "      <td>60</td>\n",
       "      <td>1</td>\n",
       "      <td>134.59</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Idade  UsoMensal  SatisfacaoCliente  ValorMensal  Plano_Basico  \\\n",
       "0     65         80                  4       174.10           0.0   \n",
       "1     49         18                  3       101.59           1.0   \n",
       "2     19         91                  4        87.93           1.0   \n",
       "3     52          0                  1        90.74           0.0   \n",
       "4     62         60                  1       134.59           1.0   \n",
       "\n",
       "   Plano_Premium  Plano_Standard  TempoContrato_Curto  TempoContrato_Longo  \\\n",
       "0            0.0             1.0                  0.0                  0.0   \n",
       "1            0.0             0.0                  0.0                  1.0   \n",
       "2            0.0             0.0                  0.0                  0.0   \n",
       "3            0.0             1.0                  0.0                  1.0   \n",
       "4            0.0             0.0                  1.0                  0.0   \n",
       "\n",
       "   TempoContrato_Medio  \n",
       "0                  1.0  \n",
       "1                  0.0  \n",
       "2                  1.0  \n",
       "3                  0.0  \n",
       "4                  0.0  "
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train_preprocessed.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Toda e qualquer transformação aplicada aos dados de treino, deve ser aplicada aos dados de teste e aos novos dados."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Removendo as colunas categóricas originais e adicionando as codificadas\n",
    "X_test_preprocessed = X_test.drop(categorical_cols, axis = 1).reset_index(drop = True)\n",
    "X_test_preprocessed = pd.concat([X_test_preprocessed, X_test_encoded], axis = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Idade</th>\n",
       "      <th>UsoMensal</th>\n",
       "      <th>SatisfacaoCliente</th>\n",
       "      <th>ValorMensal</th>\n",
       "      <th>Plano_Basico</th>\n",
       "      <th>Plano_Premium</th>\n",
       "      <th>Plano_Standard</th>\n",
       "      <th>TempoContrato_Curto</th>\n",
       "      <th>TempoContrato_Longo</th>\n",
       "      <th>TempoContrato_Medio</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>65</td>\n",
       "      <td>80</td>\n",
       "      <td>4</td>\n",
       "      <td>174.10</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>49</td>\n",
       "      <td>18</td>\n",
       "      <td>3</td>\n",
       "      <td>101.59</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>19</td>\n",
       "      <td>91</td>\n",
       "      <td>4</td>\n",
       "      <td>87.93</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>52</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>90.74</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>62</td>\n",
       "      <td>60</td>\n",
       "      <td>1</td>\n",
       "      <td>134.59</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Idade  UsoMensal  SatisfacaoCliente  ValorMensal  Plano_Basico  \\\n",
       "0     65         80                  4       174.10           0.0   \n",
       "1     49         18                  3       101.59           1.0   \n",
       "2     19         91                  4        87.93           1.0   \n",
       "3     52          0                  1        90.74           0.0   \n",
       "4     62         60                  1       134.59           1.0   \n",
       "\n",
       "   Plano_Premium  Plano_Standard  TempoContrato_Curto  TempoContrato_Longo  \\\n",
       "0            0.0             1.0                  0.0                  0.0   \n",
       "1            0.0             0.0                  0.0                  1.0   \n",
       "2            0.0             0.0                  0.0                  0.0   \n",
       "3            0.0             1.0                  0.0                  1.0   \n",
       "4            0.0             0.0                  1.0                  0.0   \n",
       "\n",
       "   TempoContrato_Medio  \n",
       "0                  1.0  \n",
       "1                  0.0  \n",
       "2                  1.0  \n",
       "3                  0.0  \n",
       "4                  0.0  "
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Visualizando as primeiras linhas dos datasets de treino e teste após pré-processamento\n",
    "X_train_preprocessed.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Idade</th>\n",
       "      <th>UsoMensal</th>\n",
       "      <th>SatisfacaoCliente</th>\n",
       "      <th>ValorMensal</th>\n",
       "      <th>Plano_Basico</th>\n",
       "      <th>Plano_Premium</th>\n",
       "      <th>Plano_Standard</th>\n",
       "      <th>TempoContrato_Curto</th>\n",
       "      <th>TempoContrato_Longo</th>\n",
       "      <th>TempoContrato_Medio</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>36</td>\n",
       "      <td>86</td>\n",
       "      <td>3</td>\n",
       "      <td>190.77</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>61</td>\n",
       "      <td>31</td>\n",
       "      <td>4</td>\n",
       "      <td>177.03</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>24</td>\n",
       "      <td>38</td>\n",
       "      <td>2</td>\n",
       "      <td>139.14</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>26</td>\n",
       "      <td>94</td>\n",
       "      <td>4</td>\n",
       "      <td>162.87</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>70</td>\n",
       "      <td>70</td>\n",
       "      <td>1</td>\n",
       "      <td>58.34</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Idade  UsoMensal  SatisfacaoCliente  ValorMensal  Plano_Basico  \\\n",
       "0     36         86                  3       190.77           0.0   \n",
       "1     61         31                  4       177.03           1.0   \n",
       "2     24         38                  2       139.14           1.0   \n",
       "3     26         94                  4       162.87           0.0   \n",
       "4     70         70                  1        58.34           0.0   \n",
       "\n",
       "   Plano_Premium  Plano_Standard  TempoContrato_Curto  TempoContrato_Longo  \\\n",
       "0            0.0             1.0                  0.0                  0.0   \n",
       "1            0.0             0.0                  0.0                  0.0   \n",
       "2            0.0             0.0                  1.0                  0.0   \n",
       "3            0.0             1.0                  1.0                  0.0   \n",
       "4            1.0             0.0                  1.0                  0.0   \n",
       "\n",
       "   TempoContrato_Medio  \n",
       "0                  1.0  \n",
       "1                  1.0  \n",
       "2                  0.0  \n",
       "3                  0.0  \n",
       "4                  0.0  "
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test_preprocessed.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### StandardScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Variáveis numéricas\n",
    "numeric_cols = X_train_preprocessed.select_dtypes(include = ['int64', 'float64']).columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Criando o StandardScaler\n",
    "scaler = StandardScaler()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Aplicando o StandardScaler às variáveis numéricas\n",
    "X_train_preprocessed[numeric_cols] = scaler.fit_transform(X_train_preprocessed[numeric_cols])\n",
    "X_test_preprocessed[numeric_cols] = scaler.transform(X_test_preprocessed[numeric_cols])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Idade</th>\n",
       "      <th>UsoMensal</th>\n",
       "      <th>SatisfacaoCliente</th>\n",
       "      <th>ValorMensal</th>\n",
       "      <th>Plano_Basico</th>\n",
       "      <th>Plano_Premium</th>\n",
       "      <th>Plano_Standard</th>\n",
       "      <th>TempoContrato_Curto</th>\n",
       "      <th>TempoContrato_Longo</th>\n",
       "      <th>TempoContrato_Medio</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.180292</td>\n",
       "      <td>1.069020</td>\n",
       "      <td>0.676895</td>\n",
       "      <td>1.130872</td>\n",
       "      <td>-0.668043</td>\n",
       "      <td>-0.720027</td>\n",
       "      <td>1.362770</td>\n",
       "      <td>-0.685994</td>\n",
       "      <td>-0.729197</td>\n",
       "      <td>1.415730</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.178225</td>\n",
       "      <td>-1.055509</td>\n",
       "      <td>-0.029255</td>\n",
       "      <td>-0.544723</td>\n",
       "      <td>1.496910</td>\n",
       "      <td>-0.720027</td>\n",
       "      <td>-0.733799</td>\n",
       "      <td>-0.685994</td>\n",
       "      <td>1.371371</td>\n",
       "      <td>-0.706349</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-1.700652</td>\n",
       "      <td>1.445952</td>\n",
       "      <td>0.676895</td>\n",
       "      <td>-0.860385</td>\n",
       "      <td>1.496910</td>\n",
       "      <td>-0.720027</td>\n",
       "      <td>-0.733799</td>\n",
       "      <td>-0.685994</td>\n",
       "      <td>-0.729197</td>\n",
       "      <td>1.415730</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.366112</td>\n",
       "      <td>-1.672308</td>\n",
       "      <td>-1.441554</td>\n",
       "      <td>-0.795450</td>\n",
       "      <td>-0.668043</td>\n",
       "      <td>-0.720027</td>\n",
       "      <td>1.362770</td>\n",
       "      <td>-0.685994</td>\n",
       "      <td>1.371371</td>\n",
       "      <td>-0.706349</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.992405</td>\n",
       "      <td>0.383688</td>\n",
       "      <td>-1.441554</td>\n",
       "      <td>0.217856</td>\n",
       "      <td>1.496910</td>\n",
       "      <td>-0.720027</td>\n",
       "      <td>-0.733799</td>\n",
       "      <td>1.457738</td>\n",
       "      <td>-0.729197</td>\n",
       "      <td>-0.706349</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Idade  UsoMensal  SatisfacaoCliente  ValorMensal  Plano_Basico  \\\n",
       "0  1.180292   1.069020           0.676895     1.130872     -0.668043   \n",
       "1  0.178225  -1.055509          -0.029255    -0.544723      1.496910   \n",
       "2 -1.700652   1.445952           0.676895    -0.860385      1.496910   \n",
       "3  0.366112  -1.672308          -1.441554    -0.795450     -0.668043   \n",
       "4  0.992405   0.383688          -1.441554     0.217856      1.496910   \n",
       "\n",
       "   Plano_Premium  Plano_Standard  TempoContrato_Curto  TempoContrato_Longo  \\\n",
       "0      -0.720027        1.362770            -0.685994            -0.729197   \n",
       "1      -0.720027       -0.733799            -0.685994             1.371371   \n",
       "2      -0.720027       -0.733799            -0.685994            -0.729197   \n",
       "3      -0.720027        1.362770            -0.685994             1.371371   \n",
       "4      -0.720027       -0.733799             1.457738            -0.729197   \n",
       "\n",
       "   TempoContrato_Medio  \n",
       "0             1.415730  \n",
       "1            -0.706349  \n",
       "2             1.415730  \n",
       "3            -0.706349  \n",
       "4            -0.706349  "
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Visualizando as primeiras linhas dos datasets de treino e teste após pré-processamento\n",
    "X_train_preprocessed.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Idade</th>\n",
       "      <th>UsoMensal</th>\n",
       "      <th>SatisfacaoCliente</th>\n",
       "      <th>ValorMensal</th>\n",
       "      <th>Plano_Basico</th>\n",
       "      <th>Plano_Premium</th>\n",
       "      <th>Plano_Standard</th>\n",
       "      <th>TempoContrato_Curto</th>\n",
       "      <th>TempoContrato_Longo</th>\n",
       "      <th>TempoContrato_Medio</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-0.635955</td>\n",
       "      <td>1.274619</td>\n",
       "      <td>-0.029255</td>\n",
       "      <td>1.516090</td>\n",
       "      <td>-0.668043</td>\n",
       "      <td>-0.720027</td>\n",
       "      <td>1.362770</td>\n",
       "      <td>-0.685994</td>\n",
       "      <td>-0.729197</td>\n",
       "      <td>1.415730</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.929775</td>\n",
       "      <td>-0.610043</td>\n",
       "      <td>0.676895</td>\n",
       "      <td>1.198580</td>\n",
       "      <td>1.496910</td>\n",
       "      <td>-0.720027</td>\n",
       "      <td>-0.733799</td>\n",
       "      <td>-0.685994</td>\n",
       "      <td>-0.729197</td>\n",
       "      <td>1.415730</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-1.387505</td>\n",
       "      <td>-0.370177</td>\n",
       "      <td>-0.735404</td>\n",
       "      <td>0.323000</td>\n",
       "      <td>1.496910</td>\n",
       "      <td>-0.720027</td>\n",
       "      <td>-0.733799</td>\n",
       "      <td>1.457738</td>\n",
       "      <td>-0.729197</td>\n",
       "      <td>-0.706349</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-1.262247</td>\n",
       "      <td>1.548752</td>\n",
       "      <td>0.676895</td>\n",
       "      <td>0.871364</td>\n",
       "      <td>-0.668043</td>\n",
       "      <td>-0.720027</td>\n",
       "      <td>1.362770</td>\n",
       "      <td>1.457738</td>\n",
       "      <td>-0.729197</td>\n",
       "      <td>-0.706349</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1.493438</td>\n",
       "      <td>0.726354</td>\n",
       "      <td>-1.441554</td>\n",
       "      <td>-1.544164</td>\n",
       "      <td>-0.668043</td>\n",
       "      <td>1.388838</td>\n",
       "      <td>-0.733799</td>\n",
       "      <td>1.457738</td>\n",
       "      <td>-0.729197</td>\n",
       "      <td>-0.706349</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Idade  UsoMensal  SatisfacaoCliente  ValorMensal  Plano_Basico  \\\n",
       "0 -0.635955   1.274619          -0.029255     1.516090     -0.668043   \n",
       "1  0.929775  -0.610043           0.676895     1.198580      1.496910   \n",
       "2 -1.387505  -0.370177          -0.735404     0.323000      1.496910   \n",
       "3 -1.262247   1.548752           0.676895     0.871364     -0.668043   \n",
       "4  1.493438   0.726354          -1.441554    -1.544164     -0.668043   \n",
       "\n",
       "   Plano_Premium  Plano_Standard  TempoContrato_Curto  TempoContrato_Longo  \\\n",
       "0      -0.720027        1.362770            -0.685994            -0.729197   \n",
       "1      -0.720027       -0.733799            -0.685994            -0.729197   \n",
       "2      -0.720027       -0.733799             1.457738            -0.729197   \n",
       "3      -0.720027        1.362770             1.457738            -0.729197   \n",
       "4       1.388838       -0.733799             1.457738            -0.729197   \n",
       "\n",
       "   TempoContrato_Medio  \n",
       "0             1.415730  \n",
       "1             1.415730  \n",
       "2            -0.706349  \n",
       "3            -0.706349  \n",
       "4            -0.706349  "
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test_preprocessed.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Modelagem"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Criando o modelo RandomForest\n",
    "dsa_modelo_v1 = RandomForestClassifier(random_state = 42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier(random_state=42)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestClassifier(random_state=42)"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Treinando o modelo\n",
    "dsa_modelo_v1.fit(X_train_preprocessed, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fazendo previsões no conjunto de teste\n",
    "y_pred = dsa_modelo_v1.predict(X_test_preprocessed)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Avaliando o modelo\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "classification_rep = classification_report(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7966666666666666"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.83      0.84      0.83       182\n",
      "           1       0.74      0.74      0.74       118\n",
      "\n",
      "    accuracy                           0.80       300\n",
      "   macro avg       0.79      0.79      0.79       300\n",
      "weighted avg       0.80      0.80      0.80       300\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_rep)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Validação Cruzada"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Criando um novo modelo RandomForest para validação cruzada\n",
    "dsa_modelo_cv = RandomForestClassifier(random_state = 42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Realizando a validação cruzada\n",
    "# Usaremos 5 folds para a validação cruzada\n",
    "cv_scores = cross_val_score(dsa_modelo_cv, X_train_preprocessed, y_train, cv = 5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.75714286, 0.76428571, 0.75      , 0.71428571, 0.70714286])"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cv_scores"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Estes resultados fornecem uma visão mais robusta do desempenho do modelo, pois a validação cruzada avalia a capacidade do modelo de generalizar para novos dados. A variação nas pontuações de acurácia entre os diferentes folds indica que o modelo pode se comportar de maneira inconsistente em diferentes subconjuntos dos dados. Isso pode ser devido a características dos dados, como desbalanceamento de classes, ou à necessidade de um ajuste mais fino dos hiperparâmetros do modelo."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<!-- Projeto Desenvolvido na Data Science Academy - www.datascienceacademy.com.br -->\n",
    "## Otimização de Hiperparâmetros\n",
    "\n",
    "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Definindo os hiperparâmetros para serem otimizados\n",
    "param_grid = {\n",
    "    'n_estimators': [50, 100, 200],   # Número de árvores\n",
    "    'max_depth': [None, 10, 20, 30],  # Profundidade máxima da árvore\n",
    "    'min_samples_split': [2, 4, 6],   # Número mínimo de amostras para dividir um nó\n",
    "    'min_samples_leaf': [1, 2, 4]     # Número mínimo de amostras exigido em um nó folha\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Criando o modelo RandomForest para a otimização\n",
    "dsa_modelo_opt = RandomForestClassifier(random_state = 42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Configurando a busca em grade com validação cruzada\n",
    "grid_search = GridSearchCV(dsa_modelo_opt, param_grid, cv = 5, scoring = 'accuracy', n_jobs = -1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GridSearchCV(cv=5, estimator=RandomForestClassifier(random_state=42), n_jobs=-1,\n",
       "             param_grid={&#x27;max_depth&#x27;: [None, 10, 20, 30],\n",
       "                         &#x27;min_samples_leaf&#x27;: [1, 2, 4],\n",
       "                         &#x27;min_samples_split&#x27;: [2, 4, 6],\n",
       "                         &#x27;n_estimators&#x27;: [50, 100, 200]},\n",
       "             scoring=&#x27;accuracy&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" ><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GridSearchCV</label><div class=\"sk-toggleable__content\"><pre>GridSearchCV(cv=5, estimator=RandomForestClassifier(random_state=42), n_jobs=-1,\n",
       "             param_grid={&#x27;max_depth&#x27;: [None, 10, 20, 30],\n",
       "                         &#x27;min_samples_leaf&#x27;: [1, 2, 4],\n",
       "                         &#x27;min_samples_split&#x27;: [2, 4, 6],\n",
       "                         &#x27;n_estimators&#x27;: [50, 100, 200]},\n",
       "             scoring=&#x27;accuracy&#x27;)</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" ><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">estimator: RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier(random_state=42)</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" ><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier(random_state=42)</pre></div></div></div></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "GridSearchCV(cv=5, estimator=RandomForestClassifier(random_state=42), n_jobs=-1,\n",
       "             param_grid={'max_depth': [None, 10, 20, 30],\n",
       "                         'min_samples_leaf': [1, 2, 4],\n",
       "                         'min_samples_split': [2, 4, 6],\n",
       "                         'n_estimators': [50, 100, 200]},\n",
       "             scoring='accuracy')"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Realizando a otimização com o conjunto de treino\n",
    "grid_search.fit(X_train_preprocessed, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Melhores parâmetros e pontuação\n",
    "best_params = grid_search.best_params_\n",
    "best_score = grid_search.best_score_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "({'max_depth': 10,\n",
       "  'min_samples_leaf': 1,\n",
       "  'min_samples_split': 2,\n",
       "  'n_estimators': 50},\n",
       " 0.7528571428571429)"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "best_params, best_score"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Versão Final do Modelo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Criando o modelo RandomForest com os melhores hiperparâmetros\n",
    "dsa_modelo_final = RandomForestClassifier(n_estimators = best_params['n_estimators'],\n",
    "                                          max_depth = best_params['max_depth'],\n",
    "                                          min_samples_split = best_params['min_samples_split'],\n",
    "                                          min_samples_leaf = best_params['min_samples_leaf'],\n",
    "                                          random_state = 42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-3 {color: black;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-3\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestClassifier(max_depth=10, n_estimators=50, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-5\" type=\"checkbox\" checked><label for=\"sk-estimator-id-5\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier(max_depth=10, n_estimators=50, random_state=42)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestClassifier(max_depth=10, n_estimators=50, random_state=42)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Treinando o modelo final com o conjunto de treino\n",
    "dsa_modelo_final.fit(X_train_preprocessed, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Previsões com dados de teste\n",
    "y_pred_final = dsa_modelo_final.predict(X_test_preprocessed)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Avaliando o modelo final no conjunto de teste\n",
    "final_accuracy = accuracy_score(y_test, y_pred_final)\n",
    "final_classification_report = classification_report(y_test, y_pred_final)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "final_accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.85      0.82      0.83       182\n",
      "           1       0.73      0.77      0.75       118\n",
      "\n",
      "    accuracy                           0.80       300\n",
      "   macro avg       0.79      0.79      0.79       300\n",
      "weighted avg       0.80      0.80      0.80       300\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(final_classification_report)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['dsa_modelo_final.pkl']"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "joblib.dump(dsa_modelo_final, 'dsa_modelo_final.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['dsa_padronizador.pkl']"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "joblib.dump(scaler, 'dsa_padronizador.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Idade</th>\n",
       "      <th>UsoMensal</th>\n",
       "      <th>SatisfacaoCliente</th>\n",
       "      <th>ValorMensal</th>\n",
       "      <th>Plano_Basico</th>\n",
       "      <th>Plano_Premium</th>\n",
       "      <th>Plano_Standard</th>\n",
       "      <th>TempoContrato_Curto</th>\n",
       "      <th>TempoContrato_Longo</th>\n",
       "      <th>TempoContrato_Medio</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.180292</td>\n",
       "      <td>1.069020</td>\n",
       "      <td>0.676895</td>\n",
       "      <td>1.130872</td>\n",
       "      <td>-0.668043</td>\n",
       "      <td>-0.720027</td>\n",
       "      <td>1.362770</td>\n",
       "      <td>-0.685994</td>\n",
       "      <td>-0.729197</td>\n",
       "      <td>1.415730</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.178225</td>\n",
       "      <td>-1.055509</td>\n",
       "      <td>-0.029255</td>\n",
       "      <td>-0.544723</td>\n",
       "      <td>1.496910</td>\n",
       "      <td>-0.720027</td>\n",
       "      <td>-0.733799</td>\n",
       "      <td>-0.685994</td>\n",
       "      <td>1.371371</td>\n",
       "      <td>-0.706349</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-1.700652</td>\n",
       "      <td>1.445952</td>\n",
       "      <td>0.676895</td>\n",
       "      <td>-0.860385</td>\n",
       "      <td>1.496910</td>\n",
       "      <td>-0.720027</td>\n",
       "      <td>-0.733799</td>\n",
       "      <td>-0.685994</td>\n",
       "      <td>-0.729197</td>\n",
       "      <td>1.415730</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.366112</td>\n",
       "      <td>-1.672308</td>\n",
       "      <td>-1.441554</td>\n",
       "      <td>-0.795450</td>\n",
       "      <td>-0.668043</td>\n",
       "      <td>-0.720027</td>\n",
       "      <td>1.362770</td>\n",
       "      <td>-0.685994</td>\n",
       "      <td>1.371371</td>\n",
       "      <td>-0.706349</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.992405</td>\n",
       "      <td>0.383688</td>\n",
       "      <td>-1.441554</td>\n",
       "      <td>0.217856</td>\n",
       "      <td>1.496910</td>\n",
       "      <td>-0.720027</td>\n",
       "      <td>-0.733799</td>\n",
       "      <td>1.457738</td>\n",
       "      <td>-0.729197</td>\n",
       "      <td>-0.706349</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Idade  UsoMensal  SatisfacaoCliente  ValorMensal  Plano_Basico  \\\n",
       "0  1.180292   1.069020           0.676895     1.130872     -0.668043   \n",
       "1  0.178225  -1.055509          -0.029255    -0.544723      1.496910   \n",
       "2 -1.700652   1.445952           0.676895    -0.860385      1.496910   \n",
       "3  0.366112  -1.672308          -1.441554    -0.795450     -0.668043   \n",
       "4  0.992405   0.383688          -1.441554     0.217856      1.496910   \n",
       "\n",
       "   Plano_Premium  Plano_Standard  TempoContrato_Curto  TempoContrato_Longo  \\\n",
       "0      -0.720027        1.362770            -0.685994            -0.729197   \n",
       "1      -0.720027       -0.733799            -0.685994             1.371371   \n",
       "2      -0.720027       -0.733799            -0.685994            -0.729197   \n",
       "3      -0.720027        1.362770            -0.685994             1.371371   \n",
       "4      -0.720027       -0.733799             1.457738            -0.729197   \n",
       "\n",
       "   TempoContrato_Medio  \n",
       "0             1.415730  \n",
       "1            -0.706349  \n",
       "2             1.415730  \n",
       "3            -0.706349  \n",
       "4            -0.706349  "
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train_preprocessed.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Author: Data Science Academy\n",
      "\n"
     ]
    }
   ],
   "source": [
    "%watermark -a \"Data Science Academy\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "#%watermark -v -m"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "#%watermark --iversions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Fim"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {
    "height": "calc(100% - 180px)",
    "left": "10px",
    "top": "150px",
    "width": "320.263px"
   },
   "toc_section_display": true,
   "toc_window_display": true
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
