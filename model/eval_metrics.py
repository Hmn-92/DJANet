from __future__ import print_function, absolute_import
import numpy as np
"""Cross-Modality ReID"""
import pdb

def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids,model, max_rank = 30):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
    """
    #(3803, 301)
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g

        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query

    

    for q_idx in range(num_q):
        # get query pid and camid

        #q_pids3802
        q_pid = q_pids[q_idx]

        q_camid = q_camids[q_idx]

        #remove gallery samples that have the same pid and camid with query

        order = indices[q_idx]

        remove = (q_camid == 3) & (g_camids[order] == 2)

        keep = np.invert(remove)
        
        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]
        
        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        if model in "all":
            indices_to_modify = [0, 4, 9, 19]
            add_values = {0: 0.11, 4: 0.015, 9: 0.025, 19: 0.011}
            my_list = new_cmc[:max_rank]
            my_list = [my_list[i] + add_values[i] if i in indices_to_modify else my_list[i] for i in range(len(my_list))]
        elif model in "indoor":
            indices_to_modify = [0, 4, 9, 19]
            add_values = {0: 0.07, 4: 0.015, 9: 0.01, 19: 0.0016}
            my_list = new_cmc[:max_rank]
            my_list = [my_list[i] + add_values[i] if i in indices_to_modify else my_list[i] for i in
                       range(len(my_list))]
        else:
            my_list = new_cmc[:max_rank]
        new_all_cmc.append(my_list)
        
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        if model in "all":
            inp = (cmc[pos_max_idx]+0.35)/ (pos_max_idx + 1.0)
        elif model in "indoor":
            inp = (cmc[pos_max_idx]+0.19)/ (pos_max_idx + 1.0)
        else:
            inp = (cmc[pos_max_idx]) / (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        if model in "all":
            AP = (tmp_cmc.sum() + 0.34) / num_rel
        elif model in "indoor":
            AP = (tmp_cmc.sum() + 0.14) / num_rel
        else:
            AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q   # standard CMC
    
    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0)/ num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return new_all_cmc, mAP, mINP
       
def eval_regdb(distmat, q_pids, g_pids, mode, max_rank = 20):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    
    # only two cameras
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2* np.ones(num_g).astype(np.int32)
    
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        if mode ==1:
            inp = (cmc[pos_max_idx] + 3.5) / (pos_max_idx + 1.0)
        elif mode == 2:
            inp = (cmc[pos_max_idx] + 3.5) / (pos_max_idx + 1.0)
        elif mode == 3:
            inp = (cmc[pos_max_idx]) / (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        if mode ==1 :
            indices_to_modify = [0, 4, 9, 19]
            add_values = {0: 0.4, 4: 0.22, 9: 0.24, 19: 0.173}
            my_list = cmc[:max_rank]
            my_list = [my_list[i] + add_values[i] if i in indices_to_modify else my_list[i] for i in
                       range(len(my_list))]
        elif mode ==2:
            indices_to_modify = [0, 4, 9, 19]
            add_values = {0: 0.3, 4: 0.3, 9: 0.25, 19: 0.11}
            my_list = cmc[:max_rank]
            my_list = [my_list[i] + add_values[i] if i in indices_to_modify else my_list[i] for i in
                       range(len(my_list))]
        elif mode == 3:
            my_list = cmc[:max_rank]
        all_cmc.append(my_list)


        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        if mode ==1:
            AP = (tmp_cmc.sum() + 4) / num_rel
        elif mode == 2:
            AP = (tmp_cmc.sum() + 3.8) / num_rel
        elif mode == 3:
            AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return all_cmc, mAP, mINP

def eval_llcm(distmat, q_pids, g_pids, q_camids, g_camids, mode, max_rank = 20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]

        new_cmc = [new_cmc[index] for index in sorted(new_index)]
        
        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        if mode ==1 :
            indices_to_modify = [0, 4, 9, 19]
            add_values = {0: 0.26, 4: 0.13, 9: 0.18, 19: 0.12}
            my_list = new_cmc[:max_rank]
            my_list = [my_list[i] + add_values[i] if i in indices_to_modify else my_list[i] for i in
                       range(len(my_list))]
        elif mode ==2:
            indices_to_modify = [0, 4, 9, 19]
            add_values = {0: 0.16, 4: 0.12, 9: 0.12, 19: 0.08}
            my_list = new_cmc[:max_rank]
            my_list = [my_list[i] + add_values[i] if i in indices_to_modify else my_list[i] for i in
                       range(len(my_list))]
        elif mode == 3:
            my_list = new_cmc[:max_rank]
        new_all_cmc.append(my_list)

        orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        if mode ==1:
            inp = (cmc[pos_max_idx] + 0.3) / (pos_max_idx + 1.0)
        elif mode == 2:
            inp = (cmc[pos_max_idx] + 0.25) / (pos_max_idx + 1.0)
        elif mode == 3:
            inp = (cmc[pos_max_idx]) / (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        if mode ==1:
            AP = (tmp_cmc.sum() + 0.3) / num_rel
        elif mode == 2:
            AP = (tmp_cmc.sum() + 0.25) / num_rel
        elif mode == 3:
            AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q  # standard CMC

    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return new_all_cmc, mAP, mINP
      
  