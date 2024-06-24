import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import cv2
import pandas as pd

class H5:
    def ReadFile(filename, type=None):
        surface_data_container = {'surface_data': None, 'base_unit': None, 'no_data_value': None}
        intensity_data_container = {'intensity_data': None, 'base_unit': None, 'no_data_value': None}
        header_data= {}
        
        def _GetHeaderData(name, obj):
            global cam_window, cam_mode, lat_res, wave_length, magnification, max_intensity, rms_spatial_dev, peak_pixel_dev
            global frame_count, scan_direction, scan_direction, scan_increment, scan_length, scan_origin, part_thickness, scan_type
            if isinstance(obj, h5py.Dataset) or isinstance(obj, h5py.Group) and 'Attributes/{' in name:
                for key, val in obj.attrs.items():
                    header_data[key] = val
                    try:                
                        cam_window = f[f'{name}'].attrs['Data Context.Window']
                    except (KeyError):
                        pass
                    try:
                        scan_type = f[f'{name}'].attrs['Data Context.Data Attributes.System Type']
                    except (KeyError):
                        pass
                    try:
                        peak_pixel_dev = f[f'{name}'].attrs['Data Context.Data Attributes.99.5% of Peak Pixel Dev:Value']
                    except (KeyError):
                        pass
                    try:
                        cam_mode = f[f'{name}'].attrs['Data Context.Data Attributes.Camera Mode']
                    except (KeyError):
                        pass
                    try:
                        lat_res = f[f'{name}'].attrs['Data Context.Lateral Resolution:Value']
                    except (KeyError):
                        pass
                    try:
                        wave_length = f[f'{name}'].attrs['Data Context.Data Attributes.Wavelength:Value']
                    except (KeyError):
                        pass
                    try:
                        magnification = f[f'{name}'].attrs['Data Context.Data Attributes.System Magnification:Value']
                    except (KeyError):
                        pass
                    try:
                        max_intensity = f[f'{name}'].attrs['Data Context.Data Attributes.Max Intensity']
                    except (KeyError):
                        pass
                    try:
                        frame_count = f[f'{name}'].attrs['Data Context.Data Attributes.Frame Count']
                    except (KeyError):
                        pass
                    try:
                        scan_direction = f[f'{name}'].attrs['Data Context.Data Attributes.Scan Direction']
                    except (KeyError):
                        pass
                    try:
                        scan_increment = f[f'{name}'].attrs['Data Context.Data Attributes.Scan Increment:Value']
                    except (KeyError):
                        pass
                    try:
                        scan_length = f[f'{name}'].attrs['Data Context.Data Attributes.Scan Length']
                    except (KeyError):
                        pass
                    try:
                        scan_origin = f[f'{name}'].attrs['Data Context.Data Attributes.Scan Origin']
                    except (KeyError):
                        pass
                    try:
                        part_thickness = f[f'{name}'].attrs['Data Attributes.Part Thickness:Value']
                    except (KeyError):
                        pass
                    try:
                        rms_spatial_dev = f[f'{name}'].attrs['Data Context.Data Attributes.RMS Spatial Dev:Value']
                    except (KeyError):
                        pass

        def _GetSurfaceData(name, obj):
            if isinstance(obj, h5py.Dataset) and 'Data/Surface' in name:
                # Retrieve the contents of the HDF5 dataset
                data = obj[()]
                attrs = {}
                for key, val in obj.attrs.items():
                    attrs[key] = val
                surface_data_container['surface_data'] = data
                surface_data_container['base_unit'] = attrs['Z Converter']['BaseUnit']
                surface_data_container['no_data_value'] = attrs['No Data']

        def _GetIntensityData(name, obj):
            if isinstance(obj, h5py.Dataset) and 'Data/Intensity' in name:
                # Retrieve the contents of the HDF5 dataset
                intensity_data = obj[()]
                attrs = {}
                for key, val in obj.attrs.items():
                    attrs[key] = val
                intensity_data_container['intensity_data'] = intensity_data
                intensity_data_container['base_unit'] = attrs['Z Converter']['BaseUnit']
                intensity_data_container['no_data_value'] = attrs['No Data']
        try:
            with h5py.File(filename, 'r') as f:
                if type == "header":
                    f.visititems(_GetHeaderData)
                    return header_data
                elif type == "surface":
                    f.visititems(_GetSurfaceData)
                    surface_data = surface_data_container['surface_data']
                    if surface_data is not None:
                        surface_data[surface_data == surface_data_container['no_data_value']] = np.nan
                    return surface_data, surface_data_container['base_unit']
                elif type == "intensity":
                    f.visititems(_GetIntensityData)
                    intensity_data = intensity_data_container['intensity_data']
                    return intensity_data, intensity_data_container['base_unit']
        except Exception as e:
            print(e)

    def GetSurfaceValues(height_values, height_unit):
        if height_values is not None:
            # Calculate peak_valley and rms_value
            peak_valley = np.nanmax(height_values) - np.nanmin(height_values)
            rms_value = np.sqrt(np.nanmean((height_values - np.nanmean(height_values))**2))

            data = {}
            good_pixels = 0
            dead_pixels = 0
            # Print each value with its pixel coordinates
            rows, cols = height_values.shape
            for i in range(rows):
                for j in range(cols):
                    data[f'Pixel ({i},{j})'] = f'{height_values[i, j]} {height_unit}'
                    # print(f"Pixel ({i}, {j}): {height_values[i, j]} {height_unit}")
                    if np.isnan(height_values[i,j]):
                        dead_pixels += 1
                    else:
                        good_pixels += 1

            # Print peak_valley and rms_value
            print("\n{0:0.3f} {2} PV, {1:0.3f} {2} RMS".format(peak_valley, rms_value, height_unit))
            print(f"Good Pixels Count: {good_pixels}")
            print(f"Dead Pixels Count: {dead_pixels}")
            return data
        else:
            print("Surface data not found.")
            return None 
            
    def GetIntensityValues(intensity_data, intensity_unit):
        intensity_values = intensity_data/255
        data = {}
        if intensity_values is not None:
            # Calculate max and mean intensity
            max_intensity = np.nanmax(intensity_values)
            mean_intensity = np.nanmean(intensity_values)

            # Print each value with its pixel coordinates
            rows, cols = intensity_values.shape
            for i in range(rows):
                for j in range(cols):
                    data[f'Pixel ({i}, {j})'] = intensity_values[i, j]
                    # print(f"Pixel ({i}, {j}): {intensity_values[i, j]} {intensity_unit}")

            # Print max and mean intensity
            print("{0:0.3f} {2} Max Intensity, {1:0.3f} {2} Mean Intensity".format(max_intensity, mean_intensity, intensity_unit))
            return data
        else:
            print("Intensity data not found.")
            return None
            
    def GetSubstrateHeight():
        # Cam_window is in the format (x_start, y_start, x_end, y_end)
        x_start, y_start, x_end, y_end = cam_window[0]
        # Calculate Height
        height_in_meters = (y_end - y_start) * lat_res[0]
        return height_in_meters

    def GetSubstrateWidth():
        # Cam_window is in the format (x_start, y_start, x_end, y_end)
        x_start, y_start, x_end, y_end = cam_window[0]
        # Calculate Height
        width_in_meters = (x_end - x_start) * lat_res[0]
        return width_in_meters

    def GetRowGoodPixel(surface_data):
        row = {}
        # Print each value with its pixel coordinates
        rows, cols = surface_data.shape
        for i in range(rows):
            count = 0
            for j in range(cols):
                if np.isnan(surface_data[i,j]):
                    continue
                else:
                    count += 1
            row[f'row {i}'] = count
        
        total = []
        height_count = 0

        good_row = {}
        for key, val in row.items():
            if val > 0:
                good_row[f'{key}'] = val
                total.append(val)
                height_count += 1
            else:
                pass
        average = np.average(total)

        print("\nNumber of Rows with Good Pixels: ", height_count)
        print("Average Pixels per Row: ", average)
        return good_row
        
    def GetFirstLastRow(surface_data):
        rows = surface_data.shape[0]
        first_last = {}

        for i in range(rows):
            first_good = None
            last_good = None
            for j in range(surface_data.shape[1]):
                if not np.isnan(surface_data[i, j]):
                    if first_good is None:
                        first_good = j
                    last_good = j
            if first_good is not None:
                first_last[f'row {i}'] = {'first_good': first_good, 'last_good': last_good}

        return first_last


    def GetColumnGoodPixel(surface_data):
        col = {}
    
        # Print each value with its pixel coordinates
        rows, cols = surface_data.shape
        for i in range(cols):
            count = 0
            for j in range(rows):
                if np.isnan(surface_data[j, i]):
                    continue
                else:
                    count += 1
            col[f'col {i}'] = count
        
        total = []
        count = 0
        good_col = {}
        for key, val in col.items():
            if val > 0:
                good_col[f'{key}'] = val
                total.append(val)
                count += 1
            else:
                pass
        average = np.average(total)

        print("\nNumber of Columns with Good Pixels: ", count)
        print("Average Pixels per Col: ", average)
        return good_col

    def GetFirstLastColumn(surface_data):
        cols = surface_data.shape[1]
        first_last = {}

        for i in range(cols):
            first_good = None
            last_good = None
            for j in range(surface_data.shape[0]):
                if not np.isnan(surface_data[j, i]):
                    if first_good is None:
                        first_good = j
                    last_good = j
            if first_good is not None:
                first_last[f'col {i}'] = {'first_good': first_good, 'last_good': last_good}
        return first_last
        # for key, val in first_last.items():
        #     print(f"{key}: First good pixel at row {val['first_good']}, Last good pixel at row {val['last_good']}")

    def GetResolution():
        #Return the wavelength in meters stored in the H5 file
        return lat_res
    
    def GetCameraWindow():
        return cam_window
    
    def GetSystemType():
        # scan_type = f[f'{name}'].attrs['Data Context.Data Attributes.System Type']
        return scan_type
    
    def GetWavelength():
        # wave_length = f[f'{name}'].attrs['Data Context.Data Attributes.Wavelength:Value']
        return wave_length
    
    def GetScanDirection():
        # scan_direction = f[f'{name}'].attrs['Data Context.Data Attributes.Scan Direction']
        return scan_direction
    
    def GetScanOrigin():
        # scan_origin = f[f'{name}'].attrs['Data Context.Data Attributes.Scan Origin']
        return scan_origin
    
    def GetCameraMode():
        # cam_mode = f[f'{name}'].attrs['Data Context.Data Attributes.Camera Mode']
        return cam_mode
    
    def GetMagnification():
        # magnification_val = f[f'{name}'].attrs['Data Context.Data Attributes.System Magnification:Value']
        return magnification
        
    def GetMaxIntensity():                
        # max_intensity = f[f'{name}'].attrs['Data Context.Data Attributes.Max Intensity']
        return max_intensity

    """ REMAINING GETTERS """
    # frame_count = f[f'{name}'].attrs['Data Context.Data Attributes.Frame Count']
    # scan_increment = f[f'{name}'].attrs['Data Context.Data Attributes.Scan Increment:Value']
    # scan_length = f[f'{name}'].attrs['Data Context.Data Attributes.Scan Length']
    # peak_pixel_dev = f[f'{name}'].attrs['Data Context.Data Attributes.99.5% of Peak Pixel Dev:Value']
    # part_thickness = f[f'{name}'].attrs['Data Attributes.Part Thickness:Value']
    # rms_spatial_dev = f[f'{name}'].attrs['Data Context.Data Attributes.RMS Spatial Dev:Value']


class TIFF:
    def ReadFile():
        print("TIFF")
        # Open the TIFF file
        tiff_path = '/Users/josuecastellanos/Documents/XROL_Image_Processing/Image Single Motor Scan 000596 Images/Image Single Motor Scan 000596 2D Image 001.Tiff'
        tiff_image = Image.open(tiff_path)

        # Convert the image to a NumPy array
        tiff_data = np.array(tiff_image)

        # Check the data type and range
        print(f'Data type: {tiff_data.dtype}')
        print(f'Min value: {tiff_data.min()}')
        print(f'Max value: {tiff_data.max()}')
        print(tiff_data.shape)

        saturated_values = tiff_data[tiff_data >= 32751]
        negative_values = tiff_data[tiff_data < 0]

        print(f'Saturated values count: {len(saturated_values)}')
        print(f'Negative values count: {len(negative_values)}')
        print('\n')
        return
    

class TXT:
    def ReadFile():
        pd.read_csv('../XROL_Image_processing/AutoColl 2303/AC_AIF2303-1.txt', header=13, delimiter='\t')
        return
    

class PNG:
    def ReadFile():
        png_paths = glob('/Users/josuecastellanos/Documents/XROL_Image_Processing/Image Single Motor Scan 000596 Images/*.Png')

        imgs = []
        for i in png_paths:
            img_cv2 = cv2.imread(i)
            img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            img_cv2_rgb_intensity_values = img_cv2_rgb / 255

            height, width, channel = img_cv2.shape

            # Flatten image intensity values
            flatten_real_intensity_values = img_cv2_rgb_intensity_values.reshape(-1, channel)

            # Generate coordinates as tuples (y, x)
            index_coords = [(y, x) for y in range(height) for x in range(width)]
            
            # Create Pandas DataFrame with coordinates as index
            df_flatten_values = pd.DataFrame(flatten_real_intensity_values, index=index_coords, columns=['R', 'G', 'B'])
        
            imgs.append(df_flatten_values)
# ________________________________________________________
        for i in png_paths:
            image = Image.open(i)
            image = image.convert('L')
            width, height = image.size
            
        data = []
        for y in range(height):
            for x in range(width):
                intensity_value = image.getpixel((x,y))
                decimal_value = intensity_value
                data.append(((x,y), decimal_value))

        df = pd.DataFrame(data, columns=['Coordinates', 'HexValue'])
        df.set_index('Coordinates', inplace=True)
# ________________________________________________________

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(img_cv2)
        plt.show()
# ________________________________________________________
        pd.Series(img_cv2.flatten()).plot(kind='hist', bins=50, title='Distribution of Pixel Values')
        plt.show()
        return
    

class Plot: 
    def plotData(data):
        return plt.pcolormesh(data)
