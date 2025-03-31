# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    z_score.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/11 10:10:38 by dsindres          #+#    #+#              #
#    Updated: 2025/03/31 13:13:04 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import TinyStatistician as ts

def zscore(x):
    if not isinstance (x, np.ndarray):
        return None
    if x.size == 0:
        return None
    tstat = ts.TinyStatistician()
    new_x = (x - tstat.mean(x)) / tstat.std(x)
    return new_x.flatten()

if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(zscore(X))
    # Sortie : 
    # array([-0.08620324,  1.2068453 , -0.86203236,  0.51721942,  0.94823559,
    # 0.17240647, -1.89647119])
    print("\n")
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    print(zscore(Y))
    # Sortie :
    # array([ 0.11267619,  1.16432067, -1.20187941,  0.37558731,  0.98904659,
    # 0.28795027, -1.72770165])
