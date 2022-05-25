from .cifar100_utils import (
    get_pretty_cifar100_type_mapping, 
    get_cifar100_type_mapping, 
    read_cifar100_pretty_labels,
    N_COARSE_LABELS as CIFAR100_N_COARSE_LABELS,
    N_FINE_LABELS as CIFAR100_N_FINE_LABELS,
    load_data as load_cifar100_data
)
from .cifar10_utils import load_data as load_cifar10_data
from .data_splitters import HoldOutFineClassesSplitStrategy, HoldOutMetaClassesSplitStrategy

from .cifar10_utils import load_images as load_cifar10_images
from .cifar100_utils import load_images as load_cifar100_images
from .mnist_utils import load_images as load_mnist_images
from .stanford_online_products_utils import load_images as load_stanford_online_products_images
from .colorectal_histology_utils import load_images as load_colorectal_histology_images
from .plant_village_utils import load_images as load_plant_village_images


image_loaders = {
    'CIFAR-10': load_cifar10_images,
    'CIFAR-100': load_cifar100_images,
    'MNIST': load_mnist_images,
    'Stanford Online Products': load_stanford_online_products_images,
    'Colorectal Histology': load_colorectal_histology_images,
    'Plant Village': load_plant_village_images,
}
