'''
Encoding Visual Attributes in Capsules for Explainable Medical Diagnoses (X-Caps)
Original Paper by Rodney LaLonde, Drew Torigian, and Ulas Bagci (https://arxiv.org/abs/1909.05926)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the functions needed to convert the LIDC-IDRI dataset to the expected format.
'''

import os
from fnmatch import filter as fnf
from glob import glob
import xml.etree.ElementTree as ET

from tqdm import tqdm
try:
    import pydicom as pydcm
except:
    import dicom as pydcm
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from PIL import Image

from utils import safe_mkdir


def create_cropped_nodules(IMG_ROOT, OUT_ROOT):
    safe_mkdir(OUT_ROOT)

    DEFAULT_PIXEL_SPACING = 0.787109
    DEFAULT_SLICE_THICKNESS = 2.5
    CROP_EXTRA_AMT = (np.sqrt(2)-1)/2

    print('Finding all xml files in LIDC-IDRI')
    matches = []
    for d1 in tqdm(sorted(os.listdir(IMG_ROOT))):
        for d2 in sorted(os.listdir(os.path.join(IMG_ROOT, d1))):
            if d2 == 'AdamsMasks':
                continue
            for d3 in sorted(os.listdir(os.path.join(IMG_ROOT, d1, d2))):
                for f in fnf(os.listdir(os.path.join(IMG_ROOT, d1, d2, d3)), '*.xml'):
                    matches.append(os.path.join(IMG_ROOT, d1, d2, d3, f))

    print('\nCreating cropped images of all nodules in LIDC-IDRI')
    for xml_file in tqdm(matches):
        # Load dicom image
        img_path = os.path.dirname(xml_file)
        dcm_imgs = []
        for dir, _, files in os.walk(img_path):
            for file in fnf(files, '*.dcm'):
                dcm_imgs.append(os.path.join(dir, file))

        # Get ref file
        RefDs = pydcm.read_file(dcm_imgs[0])

        # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
        ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(dcm_imgs))
        if int(RefDs.Rows) > 512 or int(RefDs.Columns) > 512:
            continue

        # Load spacing values (in mm)
        try:
            pixel_space = [float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1])]
        except AttributeError as e:
            if str(e) == "'FileDataset' object has no attribute 'PixelSpacing'":
                pixel_space = [DEFAULT_PIXEL_SPACING, DEFAULT_PIXEL_SPACING]
            else:
                raise NotImplementedError('Unhandled exception in pixel spacing.')

        # Load slice thickness (in mm)
        try:
            slice_thick = float(RefDs.SliceThickness)
        except AttributeError as e:
            if str(e) == "'FileDataset' object has no attribute 'SliceThickness'":
                if os.path.basename(xml_file)[:-4] == '243' or os.path.basename(xml_file)[:-4] == '244' or \
                                os.path.basename(xml_file)[:-4] == '070':
                    slice_thick = 2.5
                elif os.path.basename(xml_file)[:-4] == '135':
                    slice_thick = 2.0
                elif os.path.basename(xml_file)[:-4] == '043':
                    slice_thick = 1.8
                else:
                    slice_thick = DEFAULT_SLICE_THICKNESS
            else:
                raise NotImplementedError('Unhandled exception in slice thickness.')

        ConstPixelSpacing = (pixel_space[0], pixel_space[1], slice_thick)

        x = np.arange(0.0, (ConstPixelDims[0] + 1) * ConstPixelSpacing[0], ConstPixelSpacing[0])
        y = np.arange(0.0, (ConstPixelDims[1] + 1) * ConstPixelSpacing[1], ConstPixelSpacing[1])
        z = np.arange(0.0, (ConstPixelDims[2] + 1) * ConstPixelSpacing[2], ConstPixelSpacing[2])

        # The array is sized based on 'ConstPixelDims'
        ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

        # loop through all the DICOM files
        sop_ids = dict()
        for filenameDCM in dcm_imgs:
            # read the file
            ds = pydcm.read_file(filenameDCM)
            # store the raw image data
            ArrayDicom[:, :, dcm_imgs.index(filenameDCM)] = ds.pixel_array
            sop_ids[ds.SOPInstanceUID] = filenameDCM

        # Load attributes
        tree = ET.parse(xml_file)
        root = tree.getroot()

        unique_nodule_list = []
        curr_nodule = -1
        for s in root.findall('{http://www.nih.gov}ResponseHeader'):
            try:
                study_id = s.find('{http://www.nih.gov}StudyInstanceUID').text
            except:
                study_id = -1
        for r_num, rad in enumerate(root.findall('{http://www.nih.gov}readingSession')):
            try:
                rad_id = rad.find('{http://www.nih.gov}servicingRadiologistID').text
                if rad_id == 'anon':
                    rad_id = 'anon-{:02d}'.format(r_num)
            except:
                rad_id = -1
            for nodule in rad.findall('{http://www.nih.gov}unblindedReadNodule'):
                nodule_id = nodule.find('{http://www.nih.gov}noduleID').text
                sub = -1; ist = -1; cal = -1; sph = -1; mar = -1; lob = -1; spi = -1; tex = -1; mal = -1
                for charac in nodule.findall('{http://www.nih.gov}characteristics'):
                    try:
                        sub = int(charac.find('{http://www.nih.gov}subtlety').text)
                    except:
                        pass
                    try:
                        ist = int(charac.find('{http://www.nih.gov}internalStructure').text)
                    except:
                        pass
                    try:
                        cal = int(charac.find('{http://www.nih.gov}calcification').text)
                    except:
                        pass
                    try:
                        sph = int(charac.find('{http://www.nih.gov}sphericity').text)
                    except:
                        pass
                    try:
                        mar = int(charac.find('{http://www.nih.gov}margin').text)
                    except:
                        pass
                    try:
                        lob = int(charac.find('{http://www.nih.gov}lobulation').text)
                    except:
                        pass
                    try:
                        spi = int(charac.find('{http://www.nih.gov}spiculation').text)
                    except:
                        pass
                    try:
                        tex = int(charac.find('{http://www.nih.gov}texture').text)
                    except:
                        pass
                    try:
                        mal = int(charac.find('{http://www.nih.gov}malignancy').text)
                    except:
                        pass
                slices = []
                x_min = 999999; x_max = -9999999; y_min = 999999; y_max = -9999999
                slice_list = nodule.findall('{http://www.nih.gov}roi')
                GT = np.zeros((ConstPixelDims[0], ConstPixelDims[1], len(slice_list)), dtype=np.uint8)
                for i, roi in enumerate(slice_list):
                    z_pos = -1*float(roi.find('{http://www.nih.gov}imageZposition').text)
                    sop_id = roi.find('{http://www.nih.gov}imageSOP_UID').text
                    for edges in roi.findall('{http://www.nih.gov}edgeMap'):
                        x_pos = int(edges.find('{http://www.nih.gov}xCoord').text)
                        y_pos = int(edges.find('{http://www.nih.gov}yCoord').text)
                        GT[y_pos,x_pos, i] = 1
                        if x_pos < x_min:
                            x_min = x_pos
                        if x_pos > x_max:
                            x_max = x_pos
                        if y_pos < y_min:
                            y_min = y_pos
                        if y_pos > y_max:
                            y_max = y_pos
                    slices.append([sop_id, z_pos])
                    GT[:,:,i] = binary_fill_holes(GT[:,:,i])

                np_slices = np.asarray(slices)
                sorted_slices = np_slices[np_slices[:, 1].argsort()]
                sorted_GT = GT[:,:,np_slices[:, 1].argsort()]

                mean_x = np.mean((x_min, x_max))
                mean_y = np.mean((y_min, y_max))
                mean_z = np.mean((float(sorted_slices[0][1]), float(sorted_slices[-1][1])))
                width = abs(x_max - x_min)
                height = abs(y_max - y_min)
                depth = abs(float(sorted_slices[-1][1]) - float(sorted_slices[0][1]))
                this_nodule = -1
                matched_list = []
                for i, nod_coords in enumerate(unique_nodule_list):
                    if (abs(nod_coords[0] - mean_x) < (nod_coords[3]+width)/4 or abs(nod_coords[0] - mean_x) <= 3) and \
                       (abs(nod_coords[1] - mean_y) < (nod_coords[4]+height)/4 or abs(nod_coords[1] - mean_y) <= 3) and \
                       (abs(nod_coords[2] - mean_z) < (nod_coords[5]+depth)/4 or abs(nod_coords[2] - mean_z) <= 3*slice_thick):
                        # Check for multiple matches
                        matched_list.append([i, np.sqrt((nod_coords[0] - mean_x)**2 + (nod_coords[1] - mean_y)**2 +
                                                        (nod_coords[2] - mean_z)**2)])
                if matched_list:
                    matched_list = np.asarray(matched_list)
                    for match in matched_list[matched_list[:, 1].argsort()]:
                        if not glob(os.path.join(OUT_ROOT, '{}_{}'.format(os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(xml_file)))), study_id),
                                   'nodule_{:03d}'.format(int(match[0])), 'rad-{}_*'.format(rad_id))):
                            this_nodule = int(match[0])
                            break
                if this_nodule == -1:
                    unique_nodule_list.append([mean_x, mean_y, mean_z, width, height, depth])
                    curr_nodule += 1
                    this_nodule = curr_nodule

                out_dir = os.path.join(OUT_ROOT, '{}_{}'.format(os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(xml_file)))), study_id),
                               'nodule_{:03d}'.format(this_nodule), 'rad-{}_sub-{}_ist-{}_cal-{}_sph-{}_mar-{}_lob-{}_'
                               'spi-{}_tex-{}_mal-{}'.format(rad_id, sub, ist, cal, sph, mar, lob, spi, tex, mal))
                safe_mkdir(out_dir)

                for i, box in enumerate(sorted_slices):
                    try:
                        dcm_slice = int(dcm_imgs.index(sop_ids[box[0]]))
                    except Exception as e:
                        box[0] = correct_images(box[0])
                        if box[0] != '-1' and box[0] != '-2':
                            dcm_slice = int(dcm_imgs.index(sop_ids[box[0]]))
                        elif box[0] == '-2':
                            continue # This option is for images which cannot be corrected but are unimportant
                        else:
                            print('Unable to locate correct .dcm slice for {}: {}. Please correct by hand.'.format(out_dir, box[0]))
                            print(e)
                            continue

                    h_extra = int(height * CROP_EXTRA_AMT)
                    w_extra = int(width * CROP_EXTRA_AMT)
                    # These are to handle single pixel annotations
                    if h_extra < 2:
                        h_extra = 2
                    if w_extra < 2:
                        w_extra = 2
                    crop = ArrayDicom[y_min-h_extra:y_max+h_extra, x_min-w_extra:x_max+w_extra, dcm_slice]
                    crop_GT = sorted_GT[y_min-h_extra:y_max+h_extra, x_min-w_extra:x_max+w_extra, i]

                    try:
                        # NOTE: Make sure to change values back to int16 from uint16 when reading the images in!!!
                        im = Image.fromarray(crop.astype('<i2'))
                        im.save(os.path.join(out_dir, '{:03d}.tif').format(i))
                        gt_im = Image.fromarray(crop_GT)
                        gt_im.save(os.path.join(out_dir, 'gt_{:03d}.tif').format(i))
                    except Exception as e:
                        print('Unable to create image for: {}.'.format(out_dir))
                        print(e)

    print('\n\nCompleted creating all nodule cropped images for LIDC-IDRI dataset!')


def create_master_list(IMG_ROOT, OUT_ROOT):
    MIN_RADS = 3
    safe_mkdir(OUT_ROOT)

    num_chars_data = np.zeros((9,6))

    print('Finding all nodule files in LIDC-IDRI with >= {} radiologists\' characteristics data'.format(MIN_RADS))
    matches = []
    nodules_total = 0
    nodules_to_use = 0
    non_mal_count = 0
    mal_count = 0

    for study_dir in tqdm(sorted(os.listdir(IMG_ROOT))):
        nodule_list = sorted(os.listdir(os.path.join(IMG_ROOT, study_dir)))
        nodules_total += len(nodule_list)
        for nodule_dir in nodule_list:
            rad_dirs = sorted(os.listdir(os.path.join(IMG_ROOT, study_dir, nodule_dir)))
            rads = len(rad_dirs)
            if rads < MIN_RADS:
                continue
            temp_list = []
            temp_char_data = np.zeros((9,6))
            for rad_dir in rad_dirs:
                if not os.listdir(os.path.join(IMG_ROOT, study_dir, nodule_dir, rad_dir)):
                    rads -= 1 # Make sure there is actually image(s) for this rad
                else:
                    split_names = rad_dir.split('_')
                    all_chars = True
                    for i in range(1,10):
                        if split_names[i][-3:] == '--1':
                            all_chars = False
                        else:
                            temp_char_data[i-1][int(split_names[i][-1])-1] += 1
                    if not all_chars:
                        rads -= 1
                    else:
                        temp_list.append(os.path.join(IMG_ROOT, study_dir, nodule_dir, rad_dir))

            if rads >= MIN_RADS:
                # Compute mean mal score
                char_data_totals = np.zeros((9,rads))
                char_data_stats = np.zeros((9,2))

                for i in range(temp_char_data.shape[0]):
                    c = 0
                    for j in range(temp_char_data.shape[1]):
                        for k in range(int(temp_char_data[i,j])):
                            char_data_totals[i,c] = (j + 1)
                            c += 1
                char_data_stats[:,0] = np.mean(char_data_totals, axis=1)
                char_data_stats[:, 1] = np.std(char_data_totals, axis=1)
                mean_mal = char_data_stats[8,0]

                if mean_mal != 3.:
                    if rads > 4:
                        print('Encountered rads > 4: {}'.format(os.path.join(IMG_ROOT, study_dir, nodule_dir))) # Sanity check for nodule matching
                    if mean_mal > 3.:
                        mal_count += 1
                    else:
                        non_mal_count += 1
                    num_chars_data += temp_char_data
                    nodules_to_use += 1
                    for file_path in temp_list:
                        matches.append([file_path] + list(np.ndarray.flatten(char_data_stats)))

    print('Found {} total nodules.'.format(nodules_total))
    print('Found {} nodules with characteristics and determinable (not score 3) avg malignancy scores.'.format(nodules_to_use))
    print('{} nodules average score below 3.0, {} nodules above 3.0.'.format(non_mal_count, mal_count))

    np.savetxt(os.path.join(OUT_ROOT, 'nodule_characteristics_counts.csv'),
               np.concatenate((np.expand_dims(np.asarray(['sub', 'ist', 'cal', 'sph', 'mar', 'lob', 'spi',
               'tex', 'mal', 'Totals:']), axis=1), np.vstack((num_chars_data.astype(np.int64),
               np.asarray(['Nodules', nodules_to_use, 'Benign', non_mal_count, 'Malig', mal_count])))), axis=1),
               fmt='%s,%s,%s,%s,%s,%s,%s', delimiter=',', header="Characteristics,1,2,3,4,5,6")

    np.savetxt(os.path.join(OUT_ROOT, 'master_nodule_list.csv'), np.asarray(matches),
               fmt='%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s')


def correct_images(sname):
    # This function contains a list of images which previously required manual correction within the LIDC-IDRI Dataset.

    # Manually correct for image 0017
    if sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.305973183883758685859912046949':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.139636132253744151113715840194'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.975363198401611311891539311888':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.225900589792147134785051710110'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.153194632177600377201998652445':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.222098252047357192090439228841'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.329142402178255247031380957411':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.276070543618203204841799986172'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.109012962923033337571132618784':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.201369290021439277502674762620'
    # Manually correct for image 0365
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.112512486762301518180007539984':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.249086187399161659167414756279'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.448378396789516014605561762604':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.216758182207805904911618558070'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.303256875597167746646589593562':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.180833422259316497536094826188'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.261962165647171557143883123825':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.680655051010882131364380217685'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.428441304577336024295581627835':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.282568083479753958511921318301'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.329221218419947342986803210392':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.130147884776737463511106208477'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.162079731049618854270820976684':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.317603920309971419052997711476'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.785736194417664146622972784664':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.180833422259316497536094826188'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.652444697985639935050732394135':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.247436296095192529771061686046'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.126539353916279887972936951408':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.119827225411893011639439591713'
    # Manually correct for image 0566
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.514074599988412913277488312051':
        sname = '-2'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.292826962115570345472638642623':
        sname = '-2'
    # Manually correct for image 0659
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.294658615382614203741435957661':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.248517083496561594434577071132'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.883951945165118277793500546792':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.218208995904859324781331654067'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.227260896757583835259462034815':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.183101167412396355129144409796'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.739565975013005403715405771404':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.276779570196705787348278946110'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.244522908325555679363936146772':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.576955311322527292170312066972'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.125776849447531170933991444187':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.944721105102210115761068591710'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.156959022761131412720241221222':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.685786061228252640465903515314'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.157441085111648851876365968475':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.218208995904859324781331654067'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.137224073243631437732289379681':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.327828856516446064398338817575'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.196786590005502760794118627532':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.205919555392658132555723231924'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.271749899549008749493412118500':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.467531607505823612652093494995'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.132283326090716626749170288137':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.715346929996135559455398127585'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.292014563425807316410737237443':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.251646556878192917000905983161'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.281350813740489812658551562167':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.188830155395223944149966050821'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.316094521169588935447289217773':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.318926701435673382024116339995'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.400249926333575297612413406645':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.247894320876850135016381965868'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.381007938661788498734279329156':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.263902120137234774391883090194'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.158922411981395099005780254611':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.266102462639631998797024975317'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.153968349496782778041856013116':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.313130758239406881022967921981'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.199135326994407563129497784698':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.137821430143892810553323149499'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.328454607276840155088910752459':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.277464141419855638903368659937'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.105901452377957975094355467039':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.270022323575518362032565947858'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.230181645068532680519497368825':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.101175636735586811268012081787'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.173090595736867429956574661962':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.311147794796034131535570099457'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.426419361480558838333009902353':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.283215380710563114133061955920'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.149783315493297937843600113966':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.211773626788832944113459632641'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.333349896902589057387703875126':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.324942356299228484760469569592'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.259329619426001073359049716159':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.218894908479906137103265765511'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.204211053191501804382709873157':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.219041363289039597488091781264'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.365700870941618176907116849738':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.269625131313796127254468189745'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.174649660921460497526396207837':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.499186182774918820678569631767'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.628619440608225619886544814747':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.288908400826505634061200144991'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.165071066866482679435986323504':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.138555834428706707378735123427'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.932783428097248153076463331304':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.403741639352083297611557443868'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.232252382783080336041314614357':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.242386520761336203399531222995'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.304249587531812156369799852687':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.233668843426769210066014174740'
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.547584917033319141420515123587':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.502898181085822091725452394574'
    # Manually correct for image 0931
    elif sname == '1.3.6.1.4.1.14519.5.2.1.6279.6001.265313295605480688537936547605':
        sname = '1.3.6.1.4.1.14519.5.2.1.6279.6001.290994721708875046196354781651'
    else:
        sname = '-1'
    return sname