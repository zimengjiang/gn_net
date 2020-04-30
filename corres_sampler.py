import numpy as np
import torch
import scipy.io
import random

''' some codes from github：
    https://github.com/adambielski/siamese-triplet/blob/master/utils.py
'''
def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def sample_correspondence(known_correspondence, img1, img2, sample_size=1024):
    """known  positive correspondence given by .mat file
        sample 1024 positive correspondences and 1024 negative correspondences
        known_correspondence{'a':Nx2, 'b':Nx2}, N: number of given correspondences, 2: pixel dim, a: image_a, b:image_b
        positive correspondences can be randomly selected,
        but we want sample hard examples for negative correspondences
    """
    # read input data
    # matches_in_1 means data in correspondence in img1
    matches_in_1 = np.array(known_correspondence['a'])
    # matches_in_2 means data in correspondence in img2
    matches_in_2 = np.array(known_correspondence['b'])
    # print(matches_in_1.shape)

    # randomly select positive correspondences
    matches_in_1_random_pos, matches_in_2_random_pos = random_select_positive_matches(matches_in_1, matches_in_2,
                                                                                      num_of_pairs=sample_size)
    # store the selected positive correspondences, whose format is identical to the known_correspondence
    pos_correspondences = {'a': matches_in_1_random_pos, 'b': matches_in_2_random_pos}
    # print(pos_correspondences)

    # randomly select negative correspondences
    matches_in_1_random_neg, matches_in_2_random_neg = random_select_negative_matches(matches_in_1, matches_in_2,
                                                                                      num_of_pairs=sample_size)
    neg_correspondences_random = {'a': matches_in_1_random_neg, 'b': matches_in_2_random_neg}
    # neg_correspondences_pos = {'a': matches_in_1_random_neg, 'b': matches_in_2_random_selected_pos}
    # print(neg_correspondences_random)
    # print(neg_correspondences_pos)

    # select the hardest negative correspondences
    matches_in_1_hard_neg, matches_in_2_hard_neg = hard_select_negative_matches(matches_in_1, matches_in_2,
                                                                                num_of_pairs=1024)
    neg_correspondences_hard = {'a': matches_in_1_hard_neg, 'b': matches_in_2_hard_neg}
    # print(neg_correspondences_hard)
    '''
    return type:
        pos_correspondences: 1024 x 2
        neg_correspondences: 1024 x 2
    '''
    return pos_correspondences, neg_correspondences_random, neg_correspondences_hard


def random_select_positive_matches(matches_in_1, matches_in_2, num_of_pairs=1024):
    # check the number of correspondences
    if matches_in_1.shape[0] < num_of_pairs: # size: Batch X N X 2
        return None, None

    # generate num_of_pairs random numbers in range
    random_index = random.sample(range(0, matches_in_1.shape[0]), num_of_pairs)
    # print(random_index)

    # select samples according to the random generated index
    matches_in_1_random_selected = [matches_in_1[index] for index in random_index]
    matches_in_2_random_selected = [matches_in_2[index] for index in random_index]
    matches_in_1_random_selected = np.array(matches_in_1_random_selected)
    matches_in_2_random_selected = np.array(matches_in_2_random_selected)

    # print(matches_in_1_random_selected)
    # print(matches_in_2_random_selected)

    return matches_in_1_random_selected, matches_in_2_random_selected
def random_select_negative_matches(matches_in_1, matches_in_2, num_of_pairs=1024):
    # check the number of correspondences
    if matches_in_1.shape[0] < num_of_pairs:
        return None, None

    # generate num_of_pairs random numbers in range
    random_index = random.sample(range(0, matches_in_1.shape[0]), num_of_pairs)
    # print(random_index)

    # select samples according to the random generated index
    matches_in_1_random_selected = [matches_in_1[index] for index in random_index]
    # matches_in_2_random_selected_pos = [matches_in_2[index] for index in random_index]

    # generate random neg index which is not equal to pos index
    random_index2 = []
    for index in random_index:
        # print(index)
        random_index2.append(get_random(0, matches_in_1.shape[0] - 1, index))
    # print(random_index2)
    matches_in_2_random_selected = [matches_in_2[index2] for index2 in random_index2]

    matches_in_1_random_selected = np.array(matches_in_1_random_selected)
    matches_in_2_random_selected = np.array(matches_in_2_random_selected)
    # matches_in_2_random_selected_pos = np.array(matches_in_2_random_selected_pos)

    # print(matches_in_1_random_selected.shape)
    # print(matches_in_2_random_selected.shape)

    # return matches_in_1_random_selected, matches_in_2_random_selected, matches_in_2_random_selected_pos
    return matches_in_1_random_selected, matches_in_2_random_selected


# generate random number which is not equal to 'not_equal_num'
def get_random(a, b, not_equal_num):
    num = random.randint(a, b)
    if num == not_equal_num:
        get_random(a, b, not_equal_num)
    return num


# select the hardest negative correspondence given positive correspondence
def hard_select_negative_matches(matches_in_1, matches_in_2, num_of_pairs=1024):
    # check the number of correspondences
    if matches_in_1.shape[1] < num_of_pairs:
        return None

    # generate num_of_pairs random numbers in range
    # random_index = random.sample(range(0, matches_in_1.shape[1]), num_of_pairs)
    # # print(random_index)

    # select samples according to the random generated index
    # matches_in_1_random_selected = [matches_in_1[index] for index in random_index]
    # matches_in_2_random_selected_pos = [matches_in_2[index] for index in random_index]

    # find the hardest correspondence for each randomly selected points in matches_in-1
    # and store the index into neg_best_index2_list
    neg_best_index2_list = []
    for index1 in range(matches_in_1.shape[1]):
        d_best = 1000000
        neg_best_index2 = 1000000
        # print(index)
        for index2 in range(matches_in_2.shape[1]):
            d = np.linalg.norm(matches_in_1[index1] - matches_in_2[index2])
            if (d <= d_best) and (index2 != index1):
                d_best = d
                neg_best_index2 = index2
        neg_best_index2_list.append(neg_best_index2)

    # select samples according to neg_best_index2_list
    matches_in_2_hard_selected = [matches_in_2[index2] for index2 in neg_best_index2_list]

    # matches_in_1_random_selected = np.array(matches_in_1_random_selected)
    matches_in_2_hard_selected = np.array(matches_in_2_hard_selected)
    # matches_in_2_random_selected_pos = np.array(matches_in_2_random_selected_pos)

    # print(matches_in_1_random_selected.shape)
    # print(matches_in_2_random_selected.shape)

    # return matches_in_1_random_selected, matches_in_2_random_selected, matches_in_2_random_selected_pos
    # return matches_in_1_random_selected, matches_in_2_random_selected
    return matches_in_2_hard_selected

def corres_sampler():
    # scipy.io.savemat(data_corr, {'matches':matches, 'img1':img1, 'img2':img2})
    data = scipy.io.loadmat('all_correspondences.mat')
    # matches have the form [x, y, x', y'].
    matches = np.array(data['matches'])
    # print(matches.shape)

    # construct desired format: known_correspondence{'a':Nx2, 'b':Nx2}
    known_correspondence = {'a': matches[:, 0:2], 'b': matches[:, 2:]}
    # print(known_correspondence)

    img1 = np.array(data['img1'])
    img2 = np.array(data['img2'])
    # print(img1.shape)

    # select correspondence the output are in the format: pos_correspondences{'a':Nx2, 'b':Nx2},
    # neg_correspondences_random{'a':Nx2, 'b':Nx2}, neg_correspondences_hard{'a':Nx2, 'b':Nx2},
    pos_correspondences, neg_correspondences_random, neg_correspondences_hard = sample_correspondence(
        known_correspondence, img1, img2, sample_size=1024)

    print(pos_correspondences)
    print(neg_correspondences_random)
    print(neg_correspondences_hard)


# if __name__ == '__main__':
#     corres_sampler()