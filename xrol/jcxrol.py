import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from io import StringIO
import cv2
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot




class H5:
    def ReadFile(file_paths, type=None):
        surface_data_container = {'surface_data': [], 'base_unit': None, 'no_data_value': None}
        intensity_data_container = {'intensity_data': [], 'base_unit': None, 'no_data_value': None}
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
                surface_data_container['surface_data'].append(data)
                surface_data_container['base_unit'] = attrs['Z Converter']['BaseUnit']
                surface_data_container['no_data_value'] = attrs['No Data']

        def _GetIntensityData(name, obj):
            if isinstance(obj, h5py.Dataset) and 'Data/Intensity' in name:
                # Retrieve the contents of the HDF5 dataset
                intensity_data = obj[()]
                attrs = {}
                for key, val in obj.attrs.items():
                    attrs[key] = val
                intensity_data_container['intensity_data'].append(intensity_data)
                intensity_data_container['base_unit'] = attrs['Z Converter']['BaseUnit']
                intensity_data_container['no_data_value'] = attrs['No Data']

        for filename in file_paths:
            try:
                with h5py.File(filename, 'r') as f:
                    if type == "header":
                        f.visititems(_GetHeaderData)
                    elif type == "surface":
                        f.visititems(_GetSurfaceData)
                    elif type == "intensity":
                        f.visititems(_GetIntensityData)
            except Exception as e:
                print(e)

        if type == "header":
            return header_data
        elif type == "surface":
            combined_surface_data = surface_data_container['surface_data']
            combined_surface_data = np.array(combined_surface_data)
            combined_surface_data[combined_surface_data == surface_data_container['no_data_value']] = np.nan
            return combined_surface_data, surface_data_container['base_unit']
        
        elif type == "intensity":
            combined_intensity_data = intensity_data_container['intensity_data']
            combined_intensity_data = np.array(combined_intensity_data)
            return combined_intensity_data, intensity_data_container['base_unit']

    def GetSurfaceValues(surface_data_list, height_unit):
        all_data = []
        for height_values in surface_data_list:
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
                        if np.isnan(height_values[i, j]):
                            dead_pixels += 1
                        else:
                            good_pixels += 1

                all_data.append({
                    "data": data,
                    "peak_valley": peak_valley,
                    "rms_value": rms_value,
                    "good_pixels": good_pixels,
                    "dead_pixels": dead_pixels
                })
            else:
                print("Surface data not found.")
                all_data.append(None)
        return all_data
            
    def GetIntensityValues(intensity_data_list, intensity_unit):
        all_data = []
        for intensity_data in intensity_data_list:
            intensity_values = intensity_data / 255
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

                all_data.append({
                    "data": data,
                    "max_intensity": max_intensity,
                    "mean_intensity": mean_intensity
                })
            else:
                print("Intensity data not found.")
                all_data.append(None)
        return all_data

    def GetSubstrateLength(pixel_counts):
        lengths = [pixels * lat_res for pixels in pixel_counts]
        return lengths

    def GetTotalRows(surface_data_list):
        return [surface_data.shape[0] for surface_data in surface_data_list]
    
    def GetTotalColumns(surface_data_list):
        return [surface_data.shape[1] for surface_data in surface_data_list]
        
    def GetTotalRowsList(surface_data_list):
        all_rows = []
        for surface_data in surface_data_list:
            row = {}
            rows, cols = surface_data.shape
            for i in range(rows):
                count = 0
                for j in range(cols):
                    if np.isnan(surface_data[i, j]):
                        pass
                    else:
                        count += 1
                row[f'row {i}'] = count
            all_rows.append(row)
        return all_rows
    
    def GetTotalColumnsList(surface_data_list):
        all_cols = []
        for idx, surface_data in enumerate(surface_data_list):
            col = {}
            rows, cols = surface_data.shape
            for i in range(cols):
                count = 0
                for j in range(rows):
                    if np.isnan(surface_data[j, i]):
                        continue
                    else:
                        count += 1
                col[f'col {i}'] = count
            all_cols.append(col)
        return all_cols
    
    def GetGoodPixelList(data_list):
        all_good_pixel_lists = []
        for data in data_list:
            good_pixel_list = {}
            for key, val in data.items():
                if val == 0:
                    pass
                else:
                    good_pixel_list[key] = val
            all_good_pixel_lists.append(good_pixel_list)
        return all_good_pixel_lists

    def CleanGoodPixelList(data_list, avgs):
        all_clean_pixel_lists = []
        for data, avg in zip(data_list, avgs):
            clean_pixel_list = {}
            for key, val in data.items():
                if val < avg:
                    pass
                else:
                    clean_pixel_list[key] = val
            all_clean_pixel_lists.append(clean_pixel_list)
        return all_clean_pixel_lists

    def GetPixelAvg(data_list):
        all_avgs = []
        for data in data_list:
            total = [val for val in data.values() if val > 0]
            all_avgs.append(np.average(total))
        return all_avgs

    def GetGoodCount(data_list):
        return [len(data) for data in data_list]

    def GetFirstLastRow(surface_data_list):
        all_first_last_rows = []
        for surface_data in surface_data_list:
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
            all_first_last_rows.append(first_last)
        return all_first_last_rows

    def GetFirstLastColumn(surface_data_list):
        all_first_last_cols = []
        for surface_data in surface_data_list:
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
            all_first_last_cols.append(first_last)
        return all_first_last_cols

    def GetStandardDev(data_list):
        all_stds = []
        for data in data_list:
            std = np.std(list(data.values()))
            all_stds.append(std)
        return all_stds
    
    # def GetMiddleColumn(surface_data):
    #     # Get the dimensions of the input data
    #     num_arrays, num_rows, num_cols = surface_data.shape
        
    #     # Calculate the index for the middle column
    #     mid_col = num_cols // 2
        
    #     # Extract the middle column while keeping the 3D structure
    #     all_mids = surface_data[:, :, mid_col:mid_col+1]
        
    #     return all_mids

    def GetMiddleColumn(surface_data):
        # Initialize an array to store the middle columns
        num_arrays = surface_data.shape[0]
        num_rows = surface_data.shape[1]
        
        all_mids = np.empty((num_arrays, num_rows))
        
        for i, data in enumerate(surface_data):
            # Calculate the index for the middle column
            mid_col = data.shape[1] // 2

            # Get the middle column
            middle_column = data[:, mid_col]
            
            # Store the middle column in the array
            all_mids[i] = middle_column
            
        return all_mids
    
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
    def GlobFileNames(dirNum):
        basepath = '/Volumes/SanDisk/Image Single Motor Scan'  # My sandisk drive
        #basepath = '/Beamline Controls/BCS Setup Data/240625'
        
        dir_path = f'{basepath} 000{dirNum} Images/*.Png'
        
        # if not os.path.isfile(dir_path):
        #     print(f"File not found: {dir_path}")
        #     return
        
        file_list = glob(dir_path)  # Sort list
        file_list.sort()
        
        return file_list
    
    def ExtractData(fp):
        # Read the file content
        with open(fp, 'r') as file:
            lines = file.readlines()

        # Find the line where DATA starts
        for i, line in enumerate(lines):
            if 'DATA' in line:
                data_start_idx = i + 1
                break
        
        return data_start_idx, lines
    
    def ReadData2DF(start_index, lines):
        # Read the data into a pandas DataFrame
        data_lines = lines[start_index:]
        data_str = ''.join(data_lines)
        
        # Convert the data into a DataFrame
        data = StringIO(data_str)
        df = pd.read_csv(data, sep='\t')

        return df
    
    
    def CropImages(files, startXpixel, endXpixel, startYpixel, endYpixel, coords):
        def CropRotateImagesToRegion(file, startXpixel, endXpixel, startYpixel, endYpixel, coordinate):
            def ModifyFilename(file):
                basename = os.path.basename(file)
                # Remove a specific part of the string
                modified_basename = basename.replace('Image Single Motor Scan ', '')
                modified_basename = modified_basename.replace(' ', '_')

                return modified_basename

            # Open an image file
            with Image.open(file) as img:
                # Crop the image
                cropped_img = img.crop((startXpixel, startYpixel, endXpixel, endYpixel))
                # Rotate the image 90 degrees, Pillow 90 default is counterclockwise so -90 is needed for clockwise
                rotated_img = cropped_img.rotate(-90, expand=True)
                # Modify the new file name of image
                mod_name = ModifyFilename(file)
                # Save the cropped image
                cropped_file = f"Cropped_Rotated_{coordinate}mm_" + mod_name
                rotated_img.save(cropped_file)
                print(f"Cropped image saved as: {cropped_file}")
                
                return cropped_file

        croppedFiles = []
        for file, coordinate in zip(files, coords):
            cropped_file = CropRotateImagesToRegion(file, startXpixel, endXpixel, startYpixel, endYpixel, coordinate)
            if cropped_file:
                croppedFiles.append(cropped_file)

        return croppedFiles


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
    def PlotData(surface_data):
        fig, axs = plt.subplots(len(surface_data), 1, figsize=(8, 6 * len(surface_data)))
        
        for i, data in enumerate(surface_data):
            axs[i].pcolormesh(data)
            axs[i].set_title(f"Dataset {i+1}")
            axs[i].set_xlabel("X-axis")
            axs[i].set_ylabel("Y-axis")
            fig.colorbar(axs[i].pcolormesh(data), ax=axs[i])
        
        plt.tight_layout()
        plt.show()

class Graph:
    def GraphData(data):
        # Plot each middle column
        for i, column in enumerate(data):
            plt.figure(figsize=(10, 4))
            plt.plot(column)
            plt.title(f'Middle Column {i+1}')
            plt.xlabel('Row Index')
            plt.ylabel('Value')
            plt.grid(True)
            plt.show()

    def plot_3d_surface_multi(data):
        # Ensure data is a numpy array
        data = np.array(data)
        
        # Create subplots: 4 rows, 3 columns
        fig = make_subplots(
            rows=4, cols=3,
            specs=[[{'type': 'surface'}]*3]*4,
            subplot_titles=[f'Image {i+1}' for i in range(11)] + [''],  # 11 titles + 1 empty for the last subplot
            vertical_spacing=0.05,
            horizontal_spacing=0.05
        )

        # Create x and y coordinates
        x = np.linspace(0, data.shape[2], num=100)
        y = np.linspace(0, data.shape[1], num=100)
        x, y = np.meshgrid(x, y)

        # Add each image as a surface plot
        for i in range(11):
            row = i // 3 + 1
            col = i % 3 + 1
            
            # Downsample the data for better performance
            z = data[i, ::12, ::16]
            
            fig.add_trace(
                go.Surface(z=z, x=x, y=y, showscale=False),
                row=row, col=col
            )

        # Update the layout for better visualization
        fig.update_layout(
            title='3D Surface Plots of 11 Images',
            autosize=False,
            width=1200,
            height=1600,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Intensity'
            ),
        )

        # Update scene properties for each subplot
        for i in range(1, 12):
            fig['layout'][f'scene{i}'].update(
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5)
            )

        # Show the plot
        fig.show()


    def plot_3d_surface_multi1(data):
        # Ensure data is a numpy array
        data = np.array(data)

        # Create x and y coordinates
        x = np.linspace(0, data.shape[2], num=100)
        y = np.linspace(0, data.shape[1], num=100)
        x, y = np.meshgrid(x, y)

        # Create individual plots for each image
        for i in range(data.shape[0]):
            # Downsample the data for better performance
            z = data[i, ::12, ::16]
            
            # Create the 3D surface plot
            fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])

            # Update the layout for better visualization
            fig.update_layout(
                title=f'3D Surface Plot of Image {i+1}',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Intensity',
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=0.5)
                ),
                autosize=False,
                width=800,
                height=600,
                margin=dict(l=65, r=50, b=65, t=90)
            )

            # Show the plot
            fig.show()

    def Plot3DColumnSurfaces(data):
        # Create x and y coordinates
        x = np.arange(len(data[0]))
        y = np.arange(len(data))
        x, y = np.meshgrid(x, y)

        # Create the 3D surface plot
        fig = go.Figure(data=[go.Surface(z=data, x=x, y=y)])

        # Update the layout for better visualization
        fig.update_layout(
            title='3D Surface Plot of Data',
            scene=dict(
                xaxis_title='Column Index',
                yaxis_title='Row Index',
                zaxis_title='Value'
            ),
            autosize=False,
            width=800,
            height=600,
            margin=dict(l=65, r=50, b=65, t=90)
        )

        # Show the plot
        fig.show()

    def Plot3DMultiSurfaces(data):   
        # Create x and y coordinates
        x = np.linspace(0, data.shape[2], num=100)
        y = np.linspace(0, data.shape[1], num=100)
        x, y = np.meshgrid(x, y)

        # Create a list to store all figures
        figures = []

        # Create individual plots for each image
        for i in range(data.shape[0]):
            # Downsample the data for better performance
            z = data[i, ::12, ::16]
            
            # Create a new figure for each image
            fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
            
            # Update the layout for better visualization
            fig.update_layout(
                title=f'3D Surface Plot of Image {i+1}',
                autosize=False,
                width=600,
                height=600,
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Intensity',
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=0.5)
                ),
            )
            
            figures.append(fig)
        for fig in figures:
            fig.show()


    def Plot3DMultiSurfacesHTML(data, output_dir='3d_surface_plots'):
        # Ensure data is a numpy array
        data = np.array(data)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create x and y coordinates
        x = np.linspace(0, data.shape[2], num=100)
        y = np.linspace(0, data.shape[1], num=100)
        x, y = np.meshgrid(x, y)

        # Create individual plots for each image
        for i in range(data.shape[0]):
            # Downsample the data for better performance
            z = data[i, ::12, ::16]
            
            # Create the 3D surface plot
            fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])

            # Update the layout for better visualization
            fig.update_layout(
                title=f'3D Surface Plot of Image {i+1}',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Intensity',
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=0.5)
                ),
                autosize=False,
                width=800,
                height=600,
                margin=dict(l=65, r=50, b=65, t=90)
            )

            # Save the plot as an HTML file
            output_file = os.path.join(output_dir, f'3d_surface_plot_image_{i+1}.html')
            fig.write_html(output_file)
            print(f"Plot for Image {i+1} saved as {output_file}")