from enum import Enum


class ColumnName(str, Enum):
    n_value = 'N_Value'
    sc_value = 'SC_Value'
    st_value = 'ST_Value'
    id = 'ID'


class IDComponents(str, Enum):
    crop = 'Crop'
    tissue = 'Tissue'
    location = 'Location'
    date = 'Date'
    sample = 'Sample'


class Crop(str, Enum):
    almond_short = 'alm'
    avocado_short = 'avo'
    citrus_short = 'cit'
    vine_short = 'vin'
    almond = 'Almond'
    avocado = 'Avocado'
    citrus = 'Citrus'
    vine = 'Vine'


class CropTissue(str, Enum):
    leaf_short = 'lea'
    leaf = 'Leaf'


class Location(str, Enum):
    kedma = 'Kedma'
    meitar = 'Meitar'
    yotvata = 'Yotvata'
    kfar_menahem = 'Kfar Menahem'
    gilat = 'Gilat'
    kabri = 'Kabri'


TARGET_VARIABLES = [ColumnName.n_value.value, ColumnName.sc_value.value, ColumnName.st_value.value]
NON_FEATURE_COLUMNS = [ColumnName.id.value] + TARGET_VARIABLES

ID_MAPPING = {
    Crop.almond_short.value: Crop.almond.value,
    Crop.avocado_short.value: Crop.avocado.value,
    Crop.citrus_short.value: Crop.citrus.value,
    Crop.vine_short.value: Crop.vine.value,
    CropTissue.leaf_short.value: CropTissue.leaf.value,
    'ked': Location.kedma.value,
    'Ked': Location.kedma.value,
    'mei': Location.meitar.value,
    'yot': Location.yotvata.value,
    'Kfa': Location.kfar_menahem.value,
    'kfa': Location.kfar_menahem.value,
    'gil': Location.gilat.value,
    'glt': Location.gilat.value,
    'Gil': Location.gilat.value,
    'kab': Location.kabri.value,
    'kbr': Location.kabri.value
}
