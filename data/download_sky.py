import os
from PIL import Image
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.sdss import SDSS as AstroQuerySDSS
from astropy.io import fits
import numpy as np


# Define galaxies and sky regions to query
galaxies = [
    {"name": "Andromeda", "ra": "010.684708", "dec": "41.26906"},
    {"name": "Whirlpool", "ra": "202.469575", "dec": "47.195258"},
]

# Define sky regions with time intervals
sky_regions = [
     {"name": "Whirlpool", "ra": "202.469575", "dec": "47.195258", "start_year": 2000, "end_year": 2020},
    
]

def query_object(ra, dec, start_year=None, end_year=None):
    coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    images = []
    try:
        if start_year and end_year:
            for year in range(start_year, end_year + 1):
                # Query each year
                query = AstroQuerySDSS.query_region(coordinates=coord, data_release=year)
                if query:
                    try:
                        images.extend(AstroQuerySDSS.get_images(query))
                    except Exception as e:
                        print(f"Error processing images for year {year}: {e}")
        else:
            images = AstroQuerySDSS.get_images(coordinates=coord)
    except Exception as e:
        print(f"Error querying object at RA: {ra}, Dec: {dec}, Error: {e}")
    return images


def preprocess_and_save(images, save_path, label, year=None):
    if images is None:
        print(f"No results for {label}")
        return

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, hdul in enumerate(images):  
        data = hdul[0].data  

        # Normalize and convert to 8-bit format
        norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        img_data = (255 * norm_data).astype(np.uint8)

        # Create and save the PIL image
        pil_img = Image.fromarray(img_data)
        pil_img = pil_img.resize((224, 224))
        img_filename = f"{label}_{year}_{i}.jpg" if year else f"{label}_{i}.jpg"
        img_path = os.path.join(save_path, img_filename)
        pil_img.save(img_path)

def download_data():
    for galaxy in galaxies:
        galaxy_images = query_object(galaxy["ra"], galaxy["dec"])
        if galaxy_images and len(galaxy_images) > 0:
            preprocess_and_save(galaxy_images, './datasets/galaxies', galaxy["name"])
        else:
            print(f"No images returned for {galaxy['name']}")

    for region in sky_regions:
        for year in range(region["start_year"], region["end_year"] + 1):
            region_images = query_object(region["ra"], region["dec"], start_year=year, end_year=year)
            if region_images and len(region_images) > 0:
                preprocess_and_save(region_images, './datasets/sky_regions', region["name"], year)
            else:
                print(f"No images returned for {region['name']} in {year}")

if __name__ == "__main__":
    download_data()