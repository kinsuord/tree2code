from utils.transforms import Rescale, WordEmbedding, TreeToTensor, Vec2Word

def main():
    # some arguments
    dataset_tree_dir = './dataset/tree'
    dataset_img_dir = './dataset/img'

    dataset = Pix2TreeDataset(
            img_dir=dataset_img_dir, tree_dir=dataset_tree_dir
    )
    word_dict = get_word_dict('word_dict.npy')
    # prepare dataset
    train_data = Pix2TreeDataset(
            img_dir=dataset_img_dir, tree_dir=dataset_tree_dir,
            partition=range(int(len(dataset)*0.8)),
            tree_transform=transforms.Compose([WordEmbedding(word_dict),TreeToTensor()]),
            img_transform=transforms.Compose([Rescale(224), transforms.ToTensor()]))

    valid_data = Pix2TreeDataset(
            img_dir=dataset_img_dir, tree_dir=dataset_tree_dir,
            partition=range(int(len(dataset)*0.8), len(dataset)),
            img_transform=transforms.Compose([Rescale(224),transforms.ToTensor()]))

def train():
    pass

def load(pth_file):
    pass

def get_word_dict(dict_file, dataset):

    # get word_dict from dict_file, if not count dict form dataset and save to file
    def count_word_dict(dataset):
        word_count = {'root':0, 'end':0}
        def count_tree(tree, word_count):
            for child in tree.children:
                count_tree(child, word_count)
            if tree.value in word_count:
                word_count[tree.value] += 1
            else:
                word_count[tree.value] = 1
        
        for i in range(len(dataset)):
            count_tree(dataset[i]['tree'], word_count)
        
        word_dict = {}
        i = 0
        for key in word_count.keys():
            a = np.zeros(len(word_count))
            a[i] = 1.0
            word_dict[key] = a
            i += 1
        return word_dict

    if not os.path.exists(dict_file):
        word_dict = count_word_dict(dataset)
        np.save(dict_file, word_dict)
    else:
        word_dict = np.load(dict_file, allow_pickle=True).item()

    return word_dict

if __name__ == '__main__':
    main()
