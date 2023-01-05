from tqdm import tqdm

def stat_LCS(userId,
             dataset):


    def LCS(traj1, traj2):

        m, n = len(traj1), len(traj2)
        dp = [[0] * (n + 1) for _ in range(0, m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if traj1[i - 1] == traj2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]
    

    user_id_list = dataset.user_id_list
    traj_dict = dataset.traj_dict
    
    # Compute the average LCS.
    target_traj = traj_dict[userId][0]
    average = 0
    for id in tqdm(user_id_list, ncols=80):
        if id == userId:
            continue
        one_traj = traj_dict[id][0]
        average += LCS(one_traj, target_traj)
    average = average / (len(user_id_list) - 1)
    print("Average LCS: %f" % average)
    
    # Compute the average LCS of neighbours.
    average = 0
    for id in dataset.top_k_user_dict[userId]:
        one_traj = traj_dict[id][0]
        average += LCS(one_traj, target_traj)
    average = average / len(dataset.top_k_user_dict[userId])
    print("Average LCS of neighbours: %f" % average)        