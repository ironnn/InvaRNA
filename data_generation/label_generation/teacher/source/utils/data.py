import pandas as pd
import os
from Bio import SeqIO
from tqdm import tqdm
import time

HUMAN_FA_FILE_PATH = './data/appris_human_v2_selected.fa'
MOUSE_FA_FILE_PATH = './data/appris_mouse_v2_selected.fa'

# Biochem - Biochemical features, for both human and mouse
BIOCHEM_FILE = './biochem_and_struct_data/human_all_biochem_feature_no_len.csv'
# Struct - Structural features
# For human
HUMAN_STRUCT_UTR5_FILE = './biochem_and_struct_data/07AUG2023_seqfold_0.7.17/appris_human_v2_actual_regions_UTR5_start.fa.sec.struct.txt'
HUMAN_STRUCT_CDS_FILE = './biochem_and_struct_data/07AUG2023_seqfold_0.7.17/appris_human_v2_actual_regions_CDS_kozak_fixed.fa.sec.struct.txt'
# For mouse
MOUSE_STRUCT_UTR5_FILE = './biochem_and_struct_data/07AUG2023_seqfold_0.7.17/appris_mouse_v2_filtered_regions_UTR5_start.fa.sec.struct.txt'
MOUSE_STRUCT_CDS_FILE = './biochem_and_struct_data/07AUG2023_seqfold_0.7.17/appris_mouse_v2_filtered_regions_CDS_kozak_fixed.fa.sec.struct.txt'

def extract_lengths(name_list):
    utr5_size, cds_size, utr3_size = None, None, None

    tx_size = int(name_list[6])

    for section in name_list:
        if section.startswith("UTR5:"):
            utr5_start, utr5_end = map(int, section[len("UTR5:"):].split("-"))
            utr5_size = utr5_end - utr5_start + 1
        elif section.startswith("CDS:"):
            cds_start, cds_end = map(int, section[len("CDS:"):].split("-"))
            cds_size = cds_end - cds_start + 1
        elif section.startswith("UTR3:"):
            utr3_start, utr3_end = map(int, section[len("UTR3:"):].split("-"))
            utr3_size = utr3_end - utr3_start + 1

    if utr5_size is None:
        utr5_size = 0
    if utr3_size is None:
        utr3_size = 0
    assert utr5_size is not None and cds_size is not None and utr3_size is not None, f'name_list: {name_list}, utr5_size: {utr5_size}, cds_size: {cds_size}, utr3_size: {utr3_size}'
    assert utr5_size + cds_size + utr3_size == tx_size

    return tx_size, utr5_size, cds_size, utr3_size

def get_struct_data(data: pd.DataFrame, species='human') -> pd.DataFrame:

    utr5_struct_path = HUMAN_STRUCT_UTR5_FILE if species == 'human' else MOUSE_STRUCT_UTR5_FILE
    cds_struct_path = HUMAN_STRUCT_CDS_FILE if species == 'human' else MOUSE_STRUCT_CDS_FILE


    struct_utr5_info = pd.read_csv(utr5_struct_path, sep='\t', index_col=None)
    struct_utr5_info['SYMBOL'] = struct_utr5_info['seqname'].str.split('|').str[5].str.replace('-', '.')
    struct_utr5_info['SYMBOL'] = struct_utr5_info['SYMBOL'].apply(lambda x: 'X' + x if x[0].isdigit() else x)
    struct_utr5_info['tx_size'] = struct_utr5_info['seqname'].str.split('|').str[6].astype(int)
    struct_utr5_info = struct_utr5_info.drop(columns=['seqname'])

    struct_cds_info = pd.read_csv(cds_struct_path, sep='\t', index_col=None)
    struct_cds_info['SYMBOL'] = struct_cds_info['seqname'].str.split('|').str[5].str.replace('-', '.')
    struct_cds_info['SYMBOL'] = struct_cds_info['SYMBOL'].apply(lambda x: 'X' + x if x[0].isdigit() else x)
    struct_cds_info['tx_size'] = struct_cds_info['seqname'].str.split('|').str[6].astype(int)
    struct_cds_info = struct_cds_info.drop(columns=['seqname'])
    
    struct_info = pd.merge(struct_utr5_info, struct_cds_info, left_on=['SYMBOL', 'tx_size'], right_on=['SYMBOL', 'tx_size'], how='outer', suffixes=('_UTR5', '_CDS'))
    struct_info = struct_info.add_prefix('struct_')
    struct_info['SYMBOL'] = struct_info['struct_SYMBOL']
    struct_info['tx_size'] = struct_info['struct_tx_size']   
    struct_info = struct_info.drop(columns=['struct_SYMBOL', 'struct_tx_size'])
    
    new_data = pd.merge(data, struct_info, on=['SYMBOL', 'tx_size'], how='left')

    return new_data

def get_genes_from_appris(te_file: pd.DataFrame, species='human') -> pd.DataFrame:
    print('Get genes form appris')
    new_data_list = []

    fa_file_path = HUMAN_FA_FILE_PATH if species == 'human' else MOUSE_FA_FILE_PATH

    # missing_list = []

    with open(fa_file_path) as fa_file:
        for record in tqdm(SeqIO.parse(fa_file, 'fasta')):
            name_list = record.name.split('|')
            symbol = name_list[5]
            symbol = symbol.replace('-', '.')
            transcript_id = name_list[0]
            gene_id = name_list[1]
            if symbol[0].isdigit():
                symbol = 'X' + symbol
            if symbol in te_file.index:
                tx_size, utr5_size, cds_size, utr3_size = extract_lengths(name_list)
                new_row = {
                    'SYMBOL': symbol, 
                    'transcript_id': transcript_id,
                    'gene_id': gene_id,
                    'tx_size': tx_size,
                    'utr5_size': utr5_size, 
                    'cds_size': cds_size,
                    'utr3_size': utr3_size,
                    'tx_sequence': str(record.seq),
                }

                for bio_source in te_file.columns:
                    new_row[bio_source] = te_file.loc[symbol, bio_source]

                new_data_list.append(new_row)
    #         else:
    #             missing_list.append(symbol)

    # with open('missing_genes_in_appris.tsv', 'w') as f:
    #     for item in missing_list:
    #         f.write("%s\n" % item)
            

    new_data = pd.DataFrame(new_data_list)
    # find duplicates in SYMBOL column
    print('Duplicates: \n', new_data[new_data.duplicated(subset=['SYMBOL'], keep=False)])
    print('NaNs getting seqs: ', new_data.isna().sum().sum())

    print('new_data shape after search: ', new_data.shape)
    print('Add Struct Data')
    struct_start_time = time.time()
    new_data = get_struct_data(new_data, species=species)
    struct_end_time = time.time()
    struct_mins = int((struct_end_time - struct_start_time) / 60)
    struct_secs = int((struct_end_time - struct_start_time) % 60)
    print(f'Struct calc. time: {struct_mins} mins {struct_secs} secs')
    print('new_data shape after struct: ', new_data.shape)
    # There are some missing genes, also need to handle genes with no UTR5
    # struct_utr5_info = pd.read_csv(STRUCT_UTR5_FILE, sep='\t', index_col=0)
    # struct_cds_info = pd.read_csv(STRUCT_CDS_FILE, sep='\t', index_col=0)
    # struct_info = pd.merge(struct_utr5_info, struct_cds_info, left_index=True, right_index=True, how='outer')
    # new_data = pd.merge(new_data, struct_info, left_on='SYMBOL', right_index=True, how='left')

    return new_data


def get_data(new_data_path, new_TE_path, symbol_to_fold_path, data_dir='./data', species='human', recreate_data=False):
    if recreate_data or not os.path.exists(os.path.join(data_dir, new_data_path)):
        print('Creating new data')

        new_TE_data = pd.read_csv(os.path.join(data_dir, new_TE_path), index_col=0).transpose()
        bio_sources = [f'bio_source_{i}' for i in new_TE_data.columns]
        new_TE_data.columns = bio_sources
        new_TE_data['mean_te'] = new_TE_data[bio_sources].mean(axis=1)
        print('New TE data shape: ', new_TE_data.shape)

        symbol_to_fold_data = pd.read_csv(os.path.join(data_dir, symbol_to_fold_path), index_col=None, sep='\t')   
        print('Symbol to fold data shape: ', symbol_to_fold_data.shape)
        
        new_data = get_genes_from_appris(new_TE_data, species=species)
        print('New data shape before fold: ', new_data.shape)
        new_data = pd.merge(new_data, symbol_to_fold_data, left_on='SYMBOL', right_on='SYMBOL', how='left')

        # print(new_data.isna().sum())
        # new_data.isna().sum().to_csv(f'./{new_data_path}_nan_counts.tsv', sep='\t', index=True)
        print("Total NaN: ", new_data.isna().sum().sum())
        print('Final Data Shape: ', new_data.shape)
        # find all indicdes in new_TE_data that are not in new_data
        diff = new_TE_data[~new_TE_data.index.isin(new_data['SYMBOL'])]
        print('diff shape: ', diff.shape)

        # diff.to_csv(f'./missing_genes_in_{new_data_path}', sep='\t', index=True)
        new_data.to_csv(os.path.join(data_dir, new_data_path), sep='\t', index=False)
        return new_data

    else:
        print('Data already exists')
        return pd.read_csv(os.path.join(data_dir, new_data_path), sep='\t', index_col=None)
    

if __name__ == '__main__':
    tqdm.pandas()
    # new_TE_path = 'human_TE_cellline_all_plain.csv'
    # new_TE_path = 'human_TE_cellline_all_dedup.csv'
    # symbol_to_fold_path = 'symbol_to_fold.tsv'
    new_TE_path = 'mouse_TE_cellline_all_plain_NA.csv'
    symbol_to_fold_path = 'mouse_symbol_to_fold.tsv'
    new_data_path = f'data_with_{new_TE_path}'
    get_data(new_data_path, new_TE_path, symbol_to_fold_path, species='mouse', recreate_data=True)
