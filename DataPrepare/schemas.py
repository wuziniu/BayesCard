from DataPrepare.graph_representation import SchemaGraph, Table


def gen_job_light_imdb_schema(csv_path):
    """
    Just like the full IMDB schema but without tables that are not used in the job-light benchmark.
    """

    schema = SchemaGraph()

    # tables

    # title
    schema.add_table(Table('title', attributes=['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id',
                                                'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr',
                                                'series_years', 'md5sum'],
                           irrelevant_attributes=['episode_of_id', 'title', 'imdb_index', 'phonetic_code', 'season_nr',
                                                  'imdb_id', 'episode_nr', 'series_years', 'md5sum'],
                           no_compression=['kind_id'],
                           csv_file_location=csv_path.format('title'),
                           table_size=3486660))

    # movie_info_idx
    schema.add_table(Table('movie_info_idx', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('movie_info_idx'),
                           irrelevant_attributes=['info', 'note'],
                           no_compression=['info_type_id'],
                           table_size=3147110))

    # movie_info
    schema.add_table(Table('movie_info', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('movie_info'),
                           irrelevant_attributes=['info', 'note'],
                           no_compression=['info_type_id'],
                           table_size=24988000))

    # cast_info
    schema.add_table(Table('cast_info', attributes=['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order',
                                                    'role_id'],
                           csv_file_location=csv_path.format('cast_info'),
                           irrelevant_attributes=['nr_order', 'note', 'person_id', 'person_role_id'],
                           no_compression=['role_id'],
                           table_size=63475800))

    # movie_keyword
    schema.add_table(Table('movie_keyword', attributes=['id', 'movie_id', 'keyword_id'],
                           csv_file_location=csv_path.format('movie_keyword'),
                           no_compression=['keyword_id'],
                           table_size=7522600))

    # movie_companies
    schema.add_table(Table('movie_companies', attributes=['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                           csv_file_location=csv_path.format('movie_companies'),
                           irrelevant_attributes=['note'],
                           no_compression=['company_id', 'company_type_id'],
                           table_size=4958300))

    # relationships
    schema.add_relationship('movie_info_idx', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_info', 'movie_id', 'title', 'id')
    schema.add_relationship('cast_info', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_keyword', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_companies', 'movie_id', 'title', 'id')

    return schema


def gen_imdb_schema(csv_path):
    """
    Specifies full imdb schema. Also tables not in the job-light benchmark.
    """
    schema = SchemaGraph()

    # tables

    # title
    schema.add_table(Table('title', attributes=['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id',
                                                'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr',
                                                'series_years', 'md5sum'],
                           irrelevant_attributes=['episode_of_id'],
                           csv_file_location=csv_path.format('title'),
                           table_size=3486660))

    # movie_info_idx
    schema.add_table(Table('movie_info_idx', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('movie_info_idx'),
                           table_size=3147110))

    # movie_info
    schema.add_table(Table('movie_info', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('movie_info'),
                           table_size=24988000))

    # info_type
    schema.add_table(Table('info_type', attributes=['id', 'info'],
                           csv_file_location=csv_path.format('info_type'),
                           table_size=113))

    # cast_info
    schema.add_table(Table('cast_info', attributes=['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order',
                                                    'role_id'],
                           csv_file_location=csv_path.format('cast_info'),
                           table_size=63475800))

    # char_name
    schema.add_table(Table('char_name', attributes=['id', 'name', 'imdb_index', 'imdb_id', 'name_pcode_nf',
                                                    'surname_pcode', 'md5sum'],
                           csv_file_location=csv_path.format('char_name'),
                           table_size=4314870))

    # role_type
    schema.add_table(Table('role_type', attributes=['id', 'role'],
                           csv_file_location=csv_path.format('role_type'),
                           table_size=0))

    # complete_cast
    schema.add_table(Table('complete_cast', attributes=['id', 'movie_id', 'subject_id', 'status_id'],
                           csv_file_location=csv_path.format('complete_cast'),
                           table_size=135086))

    # comp_cast_type
    schema.add_table(Table('comp_cast_type', attributes=['id', 'kind'],
                           csv_file_location=csv_path.format('comp_cast_type'),
                           table_size=0))

    # name
    schema.add_table(Table('name', attributes=['id', 'name', 'imdb_index', 'imdb_id', 'gender', 'name_pcode_cf',
                                               'name_pcode_nf', 'surname_pcode', 'md5sum'],
                           csv_file_location=csv_path.format('name'),
                           table_size=6379740))

    # aka_name
    schema.add_table(Table('aka_name', attributes=['id', 'person_id', 'name', 'imdb_index', 'name_pcode_cf',
                                                   'name_pcode_nf', 'surname_pcode', 'md5sum'],
                           csv_file_location=csv_path.format('aka_name'),
                           table_size=1312270))

    # movie_link, is empty
    # schema.add_table(Table('movie_link', attributes=['id', 'movie_id', 'linked_movie_id', 'link_type_id'],
    #                        csv_file_location=csv_path.format('movie_link')))

    # link_type, no relationships
    # schema.add_table(Table('link_type', attributes=['id', 'link'],
    #                        csv_file_location=csv_path.format('link_type')))

    # movie_keyword
    schema.add_table(Table('movie_keyword', attributes=['id', 'movie_id', 'keyword_id'],
                           csv_file_location=csv_path.format('movie_keyword'),
                           table_size=7522600))

    # keyword
    schema.add_table(Table('keyword', attributes=['id', 'keyword', 'phonetic_code'],
                           csv_file_location=csv_path.format('keyword'),
                           table_size=236627))

    # person_info
    schema.add_table(Table('person_info', attributes=['id', 'person_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('person_info'),
                           table_size=4130210))

    # movie_companies
    schema.add_table(Table('movie_companies', attributes=['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                           csv_file_location=csv_path.format('movie_companies'),
                           table_size=4958300))

    # company_name
    schema.add_table(Table('company_name', attributes=['id', 'name', 'country_code', 'imdb_id', 'name_pcode_nf',
                                                       'name_pcode_sf', 'md5sum'],
                           csv_file_location=csv_path.format('company_name'),
                           table_size=362131))

    # company_type
    schema.add_table(Table('company_type', attributes=['id', 'kind'],
                           csv_file_location=csv_path.format('company_type'),
                           table_size=0))

    # aka_title
    schema.add_table(Table('aka_title', attributes=['id', 'movie_id', 'title', 'imdb_index', 'kind_id',
                                                    'production_year', 'phonetic_code', 'episode_of_id', 'season_nr',
                                                    'episode_nr', 'note', 'md5sum'],
                           irrelevant_attributes=['episode_of_id'],
                           csv_file_location=csv_path.format('aka_title'),
                           table_size=528268))

    # kind_type
    schema.add_table(Table('kind_type', attributes=['id', 'kind'],
                           csv_file_location=csv_path.format('kind_type'),
                           table_size=0))

    # relationships

    # title
    # omit self-join for now
    # schema.add_relationship('title', 'episode_of_id', 'title', 'id')
    schema.add_relationship('title', 'kind_id', 'kind_type', 'id')

    # movie_info_idx
    schema.add_relationship('movie_info_idx', 'info_type_id', 'info_type', 'id')
    schema.add_relationship('movie_info_idx', 'movie_id', 'title', 'id')

    # movie_info
    schema.add_relationship('movie_info', 'info_type_id', 'info_type', 'id')
    schema.add_relationship('movie_info', 'movie_id', 'title', 'id')

    # info_type, no relationships

    # cast_info
    schema.add_relationship('cast_info', 'movie_id', 'title', 'id')
    schema.add_relationship('cast_info', 'person_id', 'name', 'id')
    schema.add_relationship('cast_info', 'person_role_id', 'char_name', 'id')
    schema.add_relationship('cast_info', 'role_id', 'role_type', 'id')

    # char_name, no relationships

    # role_type, no relationships

    # complete_cast
    schema.add_relationship('complete_cast', 'movie_id', 'title', 'id')
    schema.add_relationship('complete_cast', 'status_id', 'comp_cast_type', 'id')
    schema.add_relationship('complete_cast', 'subject_id', 'comp_cast_type', 'id')

    # comp_cast_type, no relationships

    # name, no relationships

    # aka_name
    schema.add_relationship('aka_name', 'person_id', 'name', 'id')

    # movie_link, is empty
    # schema.add_relationship('movie_link', 'link_type_id', 'link_type', 'id')
    # schema.add_relationship('movie_link', 'linked_movie_id', 'title', 'id')
    # schema.add_relationship('movie_link', 'movie_id', 'title', 'id')

    # link_type, no relationships

    # movie_keyword
    schema.add_relationship('movie_keyword', 'keyword_id', 'keyword', 'id')
    schema.add_relationship('movie_keyword', 'movie_id', 'title', 'id')

    # keyword, no relationships

    # person_info
    schema.add_relationship('person_info', 'info_type_id', 'info_type', 'id')
    schema.add_relationship('person_info', 'person_id', 'name', 'id')

    # movie_companies
    schema.add_relationship('movie_companies', 'company_id', 'company_name', 'id')
    schema.add_relationship('movie_companies', 'company_type_id', 'company_type', 'id')
    schema.add_relationship('movie_companies', 'movie_id', 'title', 'id')

    # company_name, no relationships

    # company_type, no relationships

    # aka_title
    schema.add_relationship('aka_title', 'movie_id', 'title', 'id')
    schema.add_relationship('aka_title', 'kind_id', 'kind_type', 'id')

    # kind_type, no relationships

    return schema


def gen_stats_light_schema(hdf_path):
    """
    Generate the stats schema with a small subset of data.
    """

    schema = SchemaGraph()

    # tables

    # badges
    schema.add_table(Table('badges',
                           primary_key=["Id"],
                           attributes=['Id', 'UserId', 'Date'],
                           irrelevant_attributes=[],
                           no_compression=[],
                           csv_file_location=hdf_path.format('badges'),
                           drop_id_attributes=['Id', 'UserId'],
                           table_size=79851))

    # votes
    schema.add_table(Table('votes',
                           primary_key=["Id"],
                           attributes=['Id', 'PostId', 'VoteTypeId', 'CreationDate', 'UserId', 'BountyAmount'],
                           csv_file_location=hdf_path.format('votes'),
                           irrelevant_attributes=[],
                           no_compression=['VoteTypeId'],
                           drop_id_attributes=['Id', 'PostId', 'UserId'],
                           table_size=328064))

    # postHistory
    schema.add_table(Table('postHistory',
                           primary_key=["Id"],
                           attributes=['Id', 'PostHistoryTypeId', 'PostId', 'CreationDate', 'UserId'],
                           csv_file_location=hdf_path.format('postHistory'),
                           irrelevant_attributes=[],
                           no_compression=['PostHistoryTypeId'],
                           drop_id_attributes=['Id', 'PostId', 'UserId'],
                           table_size=303187))

    # posts
    schema.add_table(Table('posts',
                           primary_key=["Id"],
                           attributes=['Id', 'PostTypeId', 'CreationDate',
                                       'Score', 'ViewCount', 'OwnerUserId',
                                       'AnswerCount', 'CommentCount', 'FavoriteCount', 'LastEditorUserId'],
                           csv_file_location=hdf_path.format('posts'),
                           irrelevant_attributes=[],
                           no_compression=['PostTypeId'],
                           drop_id_attributes=['Id', 'OwnerUserId', 'LastEditorUserId'],
                           table_size=91976))

    # users
    schema.add_table(Table('users',
                           primary_key=["Id"],
                           attributes=['Id', 'Reputation', 'CreationDate', 'Views', 'UpVotes', 'DownVotes'],
                           csv_file_location=hdf_path.format('users'),
                           no_compression=[],
                           drop_id_attributes=['Id'],
                           table_size=40325))

    # comments
    schema.add_table(Table('comments',
                           primary_key=["Id"],
                           attributes=['Id', 'PostId', 'Score', 'CreationDate', 'UserId'],
                           csv_file_location=hdf_path.format('comments'),
                           irrelevant_attributes=[],
                           no_compression=[],
                           drop_id_attributes=['Id', 'PostId', 'UserId'],
                           table_size=174305))

    # postLinks
    schema.add_table(Table('postLinks',
                           primary_key=["Id"],
                           attributes=['Id', 'CreationDate', 'PostId', 'RelatedPostId', 'LinkTypeId'],
                           csv_file_location=hdf_path.format('postLinks'),
                           irrelevant_attributes=[],
                           no_compression=["LinkTypeId"],
                           drop_id_attributes=['Id', 'PostId', 'RelatedPostId'],
                           table_size=11102))

    # tags
    schema.add_table(Table('tags', attributes=['Id', 'Count', 'ExcerptPostId'],
                           csv_file_location=hdf_path.format('tags'),
                           irrelevant_attributes=[],
                           no_compression=["LinkTypeId"],
                           drop_id_attributes=['Id', 'ExcerptPostId'],
                           table_size=1032))


    # relationships
    schema.add_relationship('comments', 'PostId', 'posts', 'Id')
    schema.add_relationship('comments', 'UserId', 'users', 'Id')

    schema.add_relationship('badges', 'UserId', 'users', 'Id')

    schema.add_relationship('tags', 'ExcerptPostId', 'posts', 'Id')

    schema.add_relationship('postLinks', 'PostId', 'posts', 'Id')
    schema.add_relationship('postLinks', 'RelatedPostId', 'posts', 'Id')

    schema.add_relationship('postHistory', 'PostId', 'posts', 'Id')
    schema.add_relationship('postHistory', 'UserId', 'users', 'Id')
    schema.add_relationship('votes', 'PostId', 'posts', 'Id')
    schema.add_relationship('votes', 'UserId', 'users', 'Id')

    schema.add_relationship('posts', 'OwnerUserId', 'users', 'Id')

    return schema



def gen_DB0_schema(hdf_path):
    """
    Generate the stats schema with a small subset of data.
    """

    schema = SchemaGraph()

    # tables

    schema.add_table(Table('table0', attributes=['Id', 'attr0', 'attr1', 'attr2', 'attr3', 'attr4', 'attr5', 'attr6'],
                           csv_file_location=hdf_path + 'table0.csv',
                           no_compression=[],
                           table_size=40325))

    # posts
    schema.add_table(Table('table1', attributes=['Id', 'attr0', 'attr1', 'attr2',
                                                 'attr3', 'attr4',
                                                 'attr5', 'attr6', 'table0Id'],
                           csv_file_location=hdf_path + 'table1.csv',
                           no_compression=[],
                           table_size=91976))


    # badges
    schema.add_table(Table('table2', attributes=['Id', 'attr0', 'table0Id'],
                           no_compression=[],
                           csv_file_location=hdf_path + 'table2.csv',
                           table_size=123357))

    # votes
    schema.add_table(Table('table3', attributes=['Id', 'attr0', 'attr1', 'attr2', 'attr3', 'table1Id', 'table0Id'],
                           csv_file_location=hdf_path + 'table3.csv',
                           no_compression=[],
                           table_size=497954))

    # postHistory
    schema.add_table(Table('table4', attributes=['Id', 'attr0', 'attr1', 'attr2', 'attr3', 'attr4', 'table1Id', 'table0Id'],
                           csv_file_location=hdf_path + 'table4.csv',
                           no_compression=['PostHistoryTypeId'],
                           table_size=300919))

    # comments
    schema.add_table(Table('table5', attributes=['Id', 'attr0', 'attr1', 'attr2', 'table1Id', 'table0Id'],
                           csv_file_location=hdf_path + 'table5.csv',
                           no_compression=[],
                           table_size=153140))

    # postLinks
    schema.add_table(Table('table6', attributes=['Id', 'attr0', 'attr1', 'attr2', 'table1Id', 'table1Id2'],
                           csv_file_location=hdf_path + 'table6.csv',
                           no_compression=[],
                           table_size=19531))

    # relationships
    schema.add_relationship('table5', 'table1Id', 'table1', 'Id')
    schema.add_relationship('table5', 'table0Id', 'table0', 'Id')

    schema.add_relationship('table2', 'table0Id', 'table0', 'Id')


    schema.add_relationship('table6', 'table1Id', 'table1', 'Id')
    schema.add_relationship('table6', 'table1Id2', 'table1', 'Id')

    schema.add_relationship('table4', 'table1Id', 'table1', 'Id')
    schema.add_relationship('table4', 'table0Id', 'table0', 'Id')
    schema.add_relationship('table3', 'table1Id', 'table1', 'Id')
    schema.add_relationship('table3', 'table0Id', 'table0', 'Id')

    schema.add_relationship('table1', 'table0Id', 'table0', 'Id')

    return schema



