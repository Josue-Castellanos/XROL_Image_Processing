import h5py
import numpy as np
import matplotlib.pyplot as plt

def ReadDatxFile(filename):
    surface_data_container = {'surface_data': None, 'base_unit': None, 'no_data_value': None}
    intensity_data_container = {'intensity_data': None, 'base_unit': None, 'no_data_value': None}
    # Helper function to process each item in the HDF5 file
    
    with h5py.File(filename, 'r') as f:
        f.visititems(ProcessHeaderData)
        f.visititems(ProcessIntensityData)
        f.visititems(ProcessSurfaceData)

    surface_data = surface_data_container['surface_data']
    intensity_data = intensity_data_container['intensity_data']

    if surface_data is not None and intensity_data is not None:
        surface_data[surface_data == surface_data_container['no_data_value']] = np.nan
        
    return surface_data, surface_data_container['base_unit'], intensity_data, intensity_data_container['base_unit']
    
    
def ProcessHeaderData(name, obj):
    print(f"{name}:")
    if isinstance(obj, h5py.Dataset) and 'Attributes/{' in name:
        name = name.split('Attributes/')[1]
        
    for key, val in obj.attrs.items():
            print(f"    {key}: {val}")
            try:
                lat_res = f[f'{name}'].attrs['Data Context.Lateral Resolution:Value']
            except (KeyError):
                pass
            try:
                cam_window = f[f'{name}'].attrs['Data Context.Window']
            except (KeyError):
                pass


def ProcessSurfaceData(name, obj):
    surface_data_container = {'surface_data': None, 'base_unit': None, 'no_data_value': None}

    if isinstance(obj, h5py.Dataset) and 'Data/Surface' in name:
        # Retrieve the contents of the HDF5 dataset
        data = obj[()]
        attrs = {}
        for k, v in obj.attrs.items():
            attrs[k] = v
        surface_data_container['surface_data'] = data
        surface_data_container['base_unit'] = attrs['Z Converter']['BaseUnit']
        surface_data_container['no_data_value'] = attrs['No Data']

def ProcessIntensityData(name, obj):
    intensity_data_container = {'intensity_data': None, 'base_unit': None, 'no_data_value': None}

    if isinstance(obj, h5py.Dataset) and 'Data/Intensity' in name:
        # Retrieve the contents of the HDF5 dataset
        intensity_data = obj[()]
        attrs = {}
        for k, v in obj.attrs.items():
            attrs[k] = v
        intensity_data_container['intensity_data'] = intensity_data
        intensity_data_container['base_unit'] = attrs['Z Converter']['BaseUnit']
        intensity_data_container['no_data_value'] = attrs['No Data']


def SurfaceHeightValues(height_values, height_unit):
    if height_values is not None:
        # Calculate peak_valley and rms_value
        peak_valley = np.nanmax(height_values) - np.nanmin(height_values)
        rms_value = np.sqrt(np.nanmean((height_values - np.nanmean(height_values))**2))

        good_pixels = 0
        dead_pixels = 0
        # Print each value with its pixel coordinates
        rows, cols = height_values.shape
        for i in range(rows):
            for j in range(cols):
                # print(f"Pixel ({i}, {j}): {height_values[i, j]} {height_unit}")
                if np.isnan(height_values[i,j]):
                    dead_pixels += 1
                else:
                    good_pixels += 1

        # Print peak_valley and rms_value
        print("\n{0:0.3f} {2} PV, {1:0.3f} {2} RMS".format(peak_valley, rms_value, height_unit))
        print(f"Good Pixels Count: {good_pixels}")
        print(f"Dead Pixels Count: {dead_pixels}")
    else:
        print("Surface data not found.")
        
        
def IntensityValues(intensity_data, intensity_unit):
    intensity_values = intensity_data/255
    if intensity_values is not None:
        # Calculate max and mean intensity
        max_intensity = np.nanmax(intensity_values)
        mean_intensity = np.nanmean(intensity_values)

        # Print each value with its pixel coordinates
        rows, cols = intensity_values.shape
        intensity_dict = {}
        for i in range(rows):
            for j in range(cols):
                intensity_dict[f'Pixel ({i}, {j})'] = intensity_values[i, j]
                # print(f"Pixel ({i}, {j}): {intensity_values[i, j]} {intensity_unit}")

        # Print max and mean intensity
        print("{0:0.3f} {2} Max Intensity, {1:0.3f} {2} Mean Intensity".format(max_intensity, mean_intensity, intensity_unit))
    else:
        print("Intensity data not found.")
        
        
def SubstrateLength(cam_window, lat_res):    
    # Check if the necessary attributes were found
    if cam_window is not None and lat_res is not None:
        # Assuming cam_window is in the format (x_start, y_start, x_end, y_end)
        x_start, y_start, x_end, y_end = cam_window[0]
        
        global height_in_meters, width_in_meters
        # Perform the calculations
        height_in_meters = (y_end - y_start) * lat_res[0]
        width_in_meters = (x_end - x_start) * lat_res[0]

        # Length of the Substrate
        print(f"Height of substrate: {height_in_meters * 1000} millimeters")
        print(f"Width of substrate: {width_in_meters * 1000} millimeters")
    else:
        print("One or more necessary attributes are incorrect.")


def RowGoodPixelAvg(height_values):
    row = {}
 
    # Print each value with its pixel coordinates
    rows, cols = height_values.shape
    for i in range(rows):
        count = 0
        for j in range(cols):
            if np.isnan(height_values[i,j]):
                continue
            else:
                count += 1
        row[f'row {i}'] = count
    
    total = []
    height_count = 0

    for key, val in row.items():
        if val > 0:
            # print(f"{key}: {val}")
            total.append(val)
            height_count += 1
        else:
            pass
    average = np.average(total)

    print("\nNumber of Rows with Good Pixels: ", height_count)
    print("Average Pixels per Row: ", average)
    
    
def FirstLastGoodPixelRow(height_values):
    rows = height_values.shape[0]
    first_last = {}

    for i in range(rows):
        first_good = None
        last_good = None
        for j in range(height_values.shape[1]):
            if not np.isnan(height_values[i, j]):
                if first_good is None:
                    first_good = j
                last_good = j
        if first_good is not None:
            first_last[f'row {i}'] = {'first_good': first_good, 'last_good': last_good}

    for key, val in first_last.items():
        print(f"{key}: First good pixel at column {val['first_good']}, Last good pixel at column {val['last_good']}")


def ColumnGoodPixelAvg(height_values):
    col = {}
 
    # Print each value with its pixel coordinates
    rows, cols = height_values.shape
    for i in range(cols):
        count = 0
        for j in range(rows):
            if np.isnan(height_values[j, i]):
                continue
            else:
                count += 1
        col[f'col {i}'] = count
    
    total = []
    count = 0

    for key, val in col.items():
        if val > 0:
            # print(f"{key}: {val}")
            total.append(val)
            count += 1
        else:
            pass
    average = np.average(total)

    print("\nNumber of Columns with Good Pixels: ", count)
    print("Average Pixels per Col: ", average)
    

def FirstLastGoodPixelColumn(height_values):
    cols = height_values.shape[1]
    first_last = {}

    for i in range(cols):
        first_good = None
        last_good = None
        for j in range(height_values.shape[0]):
            if not np.isnan(height_values[j, i]):
                if first_good is None:
                    first_good = j
                last_good = j
        if first_good is not None:
            first_last[f'col {i}'] = {'first_good': first_good, 'last_good': last_good}
        

    for key, val in first_last.items():
        print(f"{key}: First good pixel at row {val['first_good']}, Last good pixel at row {val['last_good']}")
    
def GetHeaderData():
    dict = {}
    try:
        dict['Scan Device'] = scan_type
    except(NameError):
        pass
    try:
        dict['wavelength'] = wave_length
        # print(f"Wavelength Value: {wave_length * 1000} millimeters")
    except(NameError):
        pass
    try:
        print(f"99.5% of Peak Pixel Deviation: {peak_pixel_dev} Fringes")
    except(NameError):
        pass
    try:
        print(f"Magnification Value: {magnification_val}\n")
    except(NameError):
        pass
    try:
        print(f"Camera Window: {cam_window}")
    except(NameError):
        pass
    try:
        print(f"Lateral Resolution: {lat_res } meters")
    except(NameError):
        pass
    try:
        print(f"Height of substrate: {height_in_meters} meters")
    except(NameError):
        pass
    try:
        print(f"Width of substrate: {width_in_meters} meters\n")
    except(NameError):
        pass
    try:
        print(f"Camera Mode: {cam_mode} PIXELS")
    except(NameError):
        pass
    try:
        print(f"Max Intensity: {max_intensity}")
    except(NameError):
        pass
    try:
        print(f"Frame Count: {frame_count}\n")
    except(NameError):
        pass
    try:
        print(f"Scan Direction: {scan_direction}")
    except(NameError):
        pass
    try:
        print(f"Scan Length: {scan_length}") 
    except(NameError):
        pass
    try: 
        print(f"Scan Origin: {scan_origin}")
    except(NameError):
        pass
    try:
        print(f"Scan Increment Value: {scan_increment * 1000} millimeters\n")
    except(NameError):
        pass
    try:
        print(f"RMS Spatial Deviation: {rms_spatial_dev} Fringes")
    except(NameError):
        pass
    try:
        print(f"Part Thickness Value: {part_thickness * 1000} millimeters")
    except(NameError):
        pass
    
          
def plotData(data):
    return plt.pcolormesh(data)
