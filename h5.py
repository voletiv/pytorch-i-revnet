# https://github.com/fab-jul/hdf5_dataloader
import argparse
import glob
import h5py
import numpy as np
import os
import pickle
import torch

from torch.utils.data import Dataset

default_opener = lambda p_: h5py.File(p_, 'r')


class HDF5Dataset(Dataset):

    @staticmethod
    def _get_num_in_shard(shard_p, opener=default_opener):
        base_dir = os.path.dirname(shard_p)
        p_to_num_per_shard_p = os.path.join(base_dir, 'num_per_shard.pkl')
        # Speeds up filtering massively on slow file systems...
        if os.path.isfile(p_to_num_per_shard_p):
            with open(p_to_num_per_shard_p, 'rb') as f:
                p_to_num_per_shard = pickle.load(f)
                num_per_shard = p_to_num_per_shard[os.path.basename(shard_p)]
        else:
            print(f'\rh5: Opening {shard_p}... ', end='')
            try:
                with opener(shard_p) as f:
                    num_per_shard = len([key for key in list(f.keys()) if key != 'len'])
            except:
                print(f"h5: Could not open {shard_p}!")
                num_per_shard = -1
        return num_per_shard

    @staticmethod
    def check_shard_lenghts(file_ps, opener=default_opener, remove_last_hdf5=False):
        """
        Filter away the last shard, which is assumed to be smaller. this double checks that all other shards have the
        same number of entries.
        :param file_ps: list of .hdf5 files
        :param opener:
        :return: tuple (ps, num_per_shard) where
            ps = filtered file paths,
            num_per_shard = number of entries in all of the shards in `ps`
        """
        file_ps = sorted(file_ps)  # we assume that smallest shard is at the end
        if remove_last_hdf5:
            file_ps = file_ps[:-1]
        num_per_shard_prev = None
        ps = []
        for i, p in enumerate(file_ps):
            num_per_shard = HDF5Dataset._get_num_in_shard(p, opener)
            if num_per_shard == -1:
                continue
            if num_per_shard_prev is None:  # first file
                num_per_shard_prev = num_per_shard
                ps.append(p)
                continue
            if num_per_shard_prev < num_per_shard:
                raise ValueError('Expected all shards to have the same number of elements,'
                                 'except last one. Previous had {} elements, current ({}) has {}!'.format(
                                    num_per_shard_prev, p, num_per_shard))
            if num_per_shard_prev > num_per_shard:  # assuming this is the last
                is_last = i == len(file_ps) - 1
                if not is_last:
                    raise ValueError(
                            'Found shard with too few elements, and it is not the last one! {}\n'
                            'Last: {}\n'
                            'Make sure to sort file_ps before filtering.'.format(p, file_ps[-1]))
                print('Last shard: {}, has {} elements...'.format(p, num_per_shard))
            # else: # same numer as before, all good
            ps.append(p)
        assert num_per_shard_prev is not None
        return ps, num_per_shard_prev, (len(ps) - 1)*num_per_shard_prev + num_per_shard

    def __init__(self, data_dir,
                 remove_last_hdf5=False,
                 skip_shards=0,
                 shuffle_shards=False,
                 opener=default_opener,
                 seed=29):
        self.data_dir = data_dir
        self.remove_last_hdf5 = remove_last_hdf5
        self.skip_shards = skip_shards
        self.shuffle_shards = shuffle_shards
        self.opener = opener
        self.seed = seed

        self.shard_paths = sorted(glob.glob(os.path.join(self.data_dir, '*.hdf5')))
        assert len(self.shard_paths) > 0, "h5: Directory does not have any .hdf5 files! Dir: " + self.data_dir

        self.shard_ps, self.num_per_shard, self.total_num = HDF5Dataset.check_shard_lenghts(self.shard_paths, self.opener, self.remove_last_hdf5)

        # Skip shards
        assert self.skip_shards < len(self.shard_ps), "h5: Cannot skip all shards! Found " + str(len(self.shard_ps)) + " shards in " + self.data_dir + " ; len(self.shard_paths) = " + str(len(self.shard_paths)) + "; remove_last_hdf5 = " + str(self.remove_last_hdf5)
        self.shard_ps = self.shard_ps[self.skip_shards:]
        self.total_num -= self.skip_shards * self.num_per_shard

        assert len(self.shard_ps) > 0, "h5: Could not find .hdf5 files! Dir: " + self.data_dir + " ; len(self.shard_paths) = " + str(len(self.shard_paths)) + "; remove_last_hdf5 = " + str(self.remove_last_hdf5)

        self.num_of_shards = len(self.shard_ps)

        print("h5: paths", len(self.shard_ps), "; num_per_shard", self.num_per_shard, "; total", self.total_num)

        # Shuffle shards
        if self.shuffle_shards:
            np.random.seed(seed)
            if self.total_num != self.num_per_shard * self.num_of_shards:
                ps = self.shard_ps[:-1]
                np.random.shuffle(ps)
                self.shard_ps = ps + [self.shard_ps[-1]]
            else:
                np.random.shuffle(self.shard_ps)

    def __len__(self):
        return self.total_num

    def __getitem__(self, index):
        idx = index % self.total_num
        shard_idx = idx // self.num_per_shard
        idx_in_shard = str(idx % self.num_per_shard)
        # Read from shard
        with self.opener(self.shard_ps[shard_idx]) as f:
            data = torch.from_numpy(f[idx_in_shard][()]).float()
        return data


class HDF5Maker():

    def __init__(self, out_dir, num_per_shard=1000, max_shards=None, name_fmt='shard_{:04d}.hdf5', force=False, video=False):

        self.out_dir = out_dir
        self.num_per_shard = num_per_shard
        self.max_shards= max_shards
        self.name_fmt = name_fmt
        self.force = force
        self.video = video

        if os.path.isdir(self.out_dir):
            if not self.force:
                raise ValueError('{} already exists.'.format(self.out_dir))
            print('Removing *.hdf5 files from {}...'.format(self.out_dir))
            files = glob.glob(os.path.join(self.out_dir, "*.hdf5"))
            for file in files:
                os.remove(file)
        else:
            os.makedirs(self.out_dir)

        with open(os.path.join(self.out_dir, 'log'), 'w') as f:
            info_str = '\n'.join('{}={}'.format(k, v) for k, v in [
                ('out_dir', self.out_dir),
                ('num_per_shard', self.num_per_shard),
                ('max_shards', self.max_shards),
                ('name_fmt', self.name_fmt),
                ('force', self.force)])
            print(info_str)
            f.write(info_str + '\n')

        self.writer = None
        self.shard_paths = []
        self.shard_number = 0

        # To save num_of_objs in each item
        shard_idx = 0
        idx_in_shard = 0

        self.create_new_shard()

    def create_new_shard(self):

        if self.writer:
            self.writer.close()

        self.shard_number += 1

        if self.max_shards is not None and self.shard_number == self.max_shards + 1:
            print('Created {} shards, ENDING.'.format(self.max_shards))
            return

        self.shard_p = os.path.join(self.out_dir, self.name_fmt.format(self.shard_number))
        assert not os.path.exists(self.shard_p), 'Record already exists! {}'.format(self.shard_p)
        self.shard_paths.append(self.shard_p)

        print('Creating shard # {}: {}...'.format(self.shard_number, self.shard_p))
        self.writer = h5py.File(self.shard_p, 'w')
        if self.video:
            self.writer.create_group('len')

        self.count = 0

    def add_data(self, data):

        if self.video:
            self.writer['len'].create_dataset(str(self.count), data=len(data))
            self.writer.create_group(str(self.count))
            for i, frame in enumerate(data):
                self.writer[str(self.count)].create_dataset(str(i), data=frame, compression="lzf")
        else:
            # self.writer.create_dataset(str(self.count), data=data, compression="gzip", compression_opts=9)
            self.writer.create_dataset(str(self.count), data=data, compression="lzf")

        self.count += 1

        if self.count == self.num_per_shard:
            self.create_new_shard()

    def close(self):

        self.writer.close()
        assert len(self.shard_paths)

        # Writing num_per_shard.pkl
        p_to_num_per_shard = {os.path.basename(shard_p): self.num_per_shard for shard_p in self.shard_paths}
        last_shard_p = self.shard_paths[-1]
        with h5py.File(last_shard_p, 'r') as f:
            p_to_num_per_shard[os.path.basename(last_shard_p)] = len(f.keys()) - 1 * self.video

        print("Writing", os.path.join(self.out_dir, 'num_per_shard.pkl'))
        print(p_to_num_per_shard)
        with open(os.path.join(self.out_dir, 'num_per_shard.pkl'), 'wb') as f:
            pickle.dump(p_to_num_per_shard, f)


if __name__ == "__main__":

    # Make
    h5_maker = HDF5Maker('/home/voletiv/EXPERIMENTS/h5', num_per_shard=10, force=True)

    a = [torch.zeros(12, 255, 52, 52)] * 12
    for data in a:
        h5_maker.add_data(data)

    h5_maker.close()

    # Read
    h5_ds = HDF5Dataset('/home/voletiv/EXPERIMENTS/h5', remove_last_hdf5=True)
    data = h5_ds[0]

    assert torch.all(data == a[0])

