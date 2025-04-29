# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julia Stadler, Vincent Eberle and Margret Westerkamp

# Copyright(C) 2024 Max-Planck-Society

# %%

import os
import re
import subprocess
import shutil

import numpy as np
from astropy.io import fits
try:
    import ciao_contrib.runtool as rt
    from paramio import pset
except ImportError:
    print("Ciao is not sourced or installed. Therefore some operations can't be performed")
    pass

from ...messages import message_obs, message_binning, message_exposure


class ChandraObservationInformation():

    """
    Base class to provide an interface with the CXC analysis and simulation tools.

    """

    def __init__(self, obsInfo, npix_s, npix_e, fov, elim, center=None, energy_ranges=None, chips_off=()):
        """
        Initialize the ChandraObservationInformation class.

        This method sets up the interface to the CXC data and simulation tools by initializing
        the observation information and configuring the necessary parameters.

        Parameters:
        -----------
        obsInfo : dict
            A dictionary specifying the location of all required Chandra data products:
            - event_file: The L2 event file, usually found in primary and ending in _evt2.fits
            - aspect_sol: The pointing to the telescope, found in the primary data products and ending in _asol1.fits
            - bpix_file: The bad pixel file
            - mask_file: The mask file
            - data_location: The base directory where the data files are located
        npix_s : int
            Number of pixels along each spatial axis.
        npix_e : int
            Number of (logarithmic) pixels in the energy direction.
        fov : float
            Spatial extent to be considered (in arcseconds).
        elim : tuple of int
            Minimum and maximum energy to be considered in keV.
        center : tuple, optional
            RA and DEC of the image center. If None, the nominal pointing direction will be used. Default is None.
        energy_ranges : tuple, optional
            Energy ranges for energy binning. Default is None, which means
            logscale equal-width bins will be used. If energy_ranges is set elim is ignored.
        chips_off : tuple, optional
            IDs of chips that are not considered. Default is an empty tuple. BI-Chips have IDs (5, 7).

        Returns:
        --------
        None
        """

        self.obsInfo = obsInfo

        # 1. construct full file pathes
        ###############################
        for kk in ['event_file', 'bpix_file', 'aspect_sol', 'mask_file']:
            self.obsInfo[kk] = self.obsInfo['data_location']+self.obsInfo[kk]

        # 2. get information about the telescope pointing and observation duration
        ###########################################################################
        self.obsInfo['ra']       = float(rt.dmkeypar(infile=self.obsInfo['event_file'],keyword='RA_NOM',  echo=True))
        self.obsInfo['dec']      = float(rt.dmkeypar(infile=self.obsInfo['event_file'],keyword='DEC_NOM', echo=True))
        self.obsInfo['roll']     = float(rt.dmkeypar(infile=self.obsInfo['event_file'],keyword='ROLL_NOM',echo=True))
        self.obsInfo['duration'] = float(rt.dmkeypar(infile=self.obsInfo['event_file'],keyword='EXPOSURE',echo=True))

        # 2.a) pointing direction in RA and DEC
        #      this should be identical to the nominal pointing direction calculated above
        rt.dmcoords.punlearn()
        rt.dmcoords(self.obsInfo['event_file'],
                    op='msc',
                    theta=0.0,
                    phi=0.0,
                    celfmt='deg',
                    asol=self.obsInfo['aspect_sol'])
        self.obsInfo['aim_ra']  = float(rt.dmcoords.ra)
        self.obsInfo['aim_dec'] = float(rt.dmcoords.dec)


        # 3. define discretization
        ##########################
        # externally we work with celecstial coordinates but internally with SKY coordinates
        # this makes it easier to deal with the event files which only give sky coordinates
        # for Chandra coordinate system see https://cxc.harvard.edu/contrib/jcm/ncoords.ps

        # 3.a) define the center of the image in celestial coordinates
        ra_center  = self.obsInfo['aim_ra']
        dec_center = self.obsInfo['aim_dec']
        if center!=None:
            ra_center  = center[0]
            dec_center = center[1]

        # 3.b) convert the image center to sky coordinates
        rt.dmcoords.punlearn()
        rt.dmcoords(self.obsInfo['event_file'],
                    op='cel',
                    ra=ra_center,
                    dec=dec_center,
                    celfmt='deg',
                    asol=self.obsInfo['aspect_sol'])
        self.obsInfo['x_center'] = float(rt.dmcoords.x)
        self.obsInfo['y_center'] = float(rt.dmcoords.y)

        # 3.c) define range in x and y coordinates
        # note: pixelsize = 0.492 arcsec
        # FIXME really?
        self.obsInfo['xy_range'] = fov/2/0.492  # full fov / 2 (for half fov) / 0.492 (pixel size)
        self.obsInfo['x_min']    = self.obsInfo['x_center'] - self.obsInfo['xy_range'] 
        self.obsInfo['x_max']    = self.obsInfo['x_center'] + self.obsInfo['xy_range']
        self.obsInfo['y_min']    = self.obsInfo['y_center'] - self.obsInfo['xy_range']
        self.obsInfo['y_max']    = self.obsInfo['y_center'] + self.obsInfo['xy_range']
        self.obsInfo['npix_s']   = npix_s
        self.obsInfo['fov']     = fov  # spatial pixel scale in arcsec

        # 3.d) energy discretization
        self.obsInfo['energy_ranges'] = energy_ranges
        self.obsInfo['energy_min'] = elim[0]
        self.obsInfo['energy_max'] = elim[1]
        self.obsInfo['npix_e']     = npix_e

        # 4. some information from the methods
        #####################################
        # get_data
        self.obsInfo['ntot_binned']       = 0.
        # get_exposure
        self.obsInfo['asphist_res_xy']    = None
        self.obsInfo['exp_ebins_per_bin'] = None
        self.obsInfo['chips_on']          = None
        self.obsInfo['chips_in']          = None
        self.obsInfo['chips_off'] = chips_off
        # get_psf_sim
        self.psf_sim_coords               = []

        # 5. print some information
        ###########################
        message_obs(obsInfo)

    def get_data(self, outfile):

        """
        Obtain the observed photon counts on a 3D grid over spatial and energy coordinates.

        Parameters:
        -----------
        outfile: string
            fits file to which CXC saves the filtered event list

        Returns:
        --------    
        data: np.array
            event counts on a 3D grid

        """

        # filter w/ cxc: spatial and energy cuts
        # creates an event list w/ only those detections that make the cuts
        infile  = self.obsInfo['event_file']
        infile += '[x={0:.6f}:{1:.6f},'.format(self.obsInfo['x_min'], self.obsInfo['x_max'])
        infile += 'y={0:.6f}:{1:.6f},'.format(self.obsInfo['y_min'], self.obsInfo['y_max'])
        infile += 'energy={0:.4f}:{1:.4f}]'.format(1.e3*self.obsInfo['energy_min'], 1.e3*self.obsInfo['energy_max'])

        rt.dmcopy(infile=infile, outfile=outfile, clobber='yes')

        # bin events on a numpy array
        # see https://cxc.cfa.harvard.edu/ciao/data_products_guide/events.html for L2 event file columns
        with fits.open(outfile) as dat_filtered:
            evts = dat_filtered['EVENTS'].data
        evts = np.array([evts['x'], evts['y'], np.log(1.e-3*evts['energy'])])
        evts = evts.transpose()

        if self.obsInfo['energy_ranges'] is not None:
            bins = (self.obsInfo['npix_s'],  self.obsInfo['npix_s'], np.log(self.obsInfo['energy_ranges']))
        else:
            bins = (self.obsInfo['npix_s'],  self.obsInfo['npix_s'], self.obsInfo['npix_e'])

        ranges = ((self.obsInfo['x_min'], self.obsInfo['x_max']),
                  (self.obsInfo['y_min'], self.obsInfo['y_max']),
                  (np.log(self.obsInfo['energy_min']), np.log(self.obsInfo['energy_max'])))

        data, edges = np.histogramdd(evts, bins=bins, range=ranges, density=False, weights=None)
        data = data.transpose((1, 0, 2)).astype(int)
        self.obsInfo['ntot_binned'] = np.sum(data)

        message_binning(self.obsInfo)
        if self.obsInfo["energy_ranges"] is not None:
            energy_ranges = np.log(self.obsInfo['energy_ranges'])
            print(f'Generated data for log energy ranges {energy_ranges}')
            # TODO Print edges regardless of method
        return data

    def get_exposure(self, outroot, res_xy=0.5, energy_subbins=10):
        """
        Obtain the exposure of the observation over the full fov.

        Parameters:
        -----------
        outroot: string
            file path to which the temporary CXC products are saved
        res_xy: float
            resolution in x and y for the aspect histogramm in arcsec
            (0.5 arcsex is about 1 pixel and the CXC default).
        energy_subbins: int
            energy sub-binning to compute the instrumnt map (see below).

        Returns:
        --------
        expmap: (np.array) 
            npix_e x npix_s x npix_s array with the exposure in units of  [sec * cm**(2) counts/photon]
        """

        self.obsInfo['asphist_res_xy']    = res_xy
        self.obsInfo['exp_ebins_per_bin'] = energy_subbins

        # set ardlib to use the right badpixel file
        ###########################################
        rt.ardlib.punlearn()
        os.system('acis_set_ardlib ' + self.obsInfo['bpix_file'] + ' > /dev/null')
        print(' ')

        # get all chips that are on
        ##############################
        det_str = rt.dmkeypar(infile=self.obsInfo['event_file'], keyword='detnam', echo=True)
        chips_on = [int(d) for d in re.findall(r'\d', det_str)]
        for i in range(len(self.obsInfo['chips_off'])):
            if self.obsInfo['chips_off'] in chips_on:
                chips_on.remove(self.obsInfo['chips_off'])
        chips_on = np.array(chips_on)
        # which chips fall into our region of interest
        ##############################################
        det_mask = np.full((len(chips_on)), False, dtype=bool)
        for chip in range(0, len(chips_on)):
            edgex = []
            edgey = []

            for chipx in [0.5, 1024.5]:
                for chipy in [0.5, 1024.5]:

                    rt.dmcoords.punlearn()
                    rt.dmcoords(self.obsInfo['event_file'],
                                asol=self.obsInfo['aspect_sol'],
                                opt='chip', chip_id=chips_on[chip],
                                chipx=chipx, chipy=chipy)
                    edgex += [rt.dmcoords.x]
                    edgey += [rt.dmcoords.y]

            xcrit = (self.obsInfo['x_min'] <= min(edgex) <= self.obsInfo['x_max']) or\
                    (self.obsInfo['x_min'] <= max(edgex) <= self.obsInfo['x_max']) or\
                    (min(edgex) < self.obsInfo['x_min'] and max(edgex) > self.obsInfo['x_max'])
            ycrit = (self.obsInfo['y_min'] <= min(edgey) <= self.obsInfo['y_max']) or\
                    (self.obsInfo['y_min'] <= max(edgey) <= self.obsInfo['y_max']) or\
                    (min(edgey) < self.obsInfo['y_min'] and max(edgey) > self.obsInfo['y_max'])
            det_mask[chip] = (xcrit and ycrit)

        det_num = chips_on[det_mask]

        self.obsInfo['chips_on'] = chips_on
        self.obsInfo['chips_in'] = det_num

        # compute the aspect histogramm
        ###############################
        # the aspect solution is given every 0.256 s during an observation and can be
        # represented in a compressed form as a
        # histogramm of the pointing vs. x-offset, y-offset, and roll-offset
        # see https://cxc.cfa.harvard.edu/ciao/ahelp/asphist.html
        rt.asphist.punlearn()
        asphist_dic = {}

        for det in det_num:
            outf = outroot + '_acis-{:d}.asphis'.format(det)
            asphist_dic[det] = outf
            evtf = self.obsInfo['event_file'] + '[ccd_id={:d}]'.format(det)
            rt.asphist(infile=self.obsInfo['aspect_sol'], outfile=outf, evtfile=evtf,
                       res_xy=self.obsInfo['asphist_res_xy'], clobber='yes')

        # from here on things depend on energy
        # energy bins are labeld by their left edge
        ###########################################
        logemin = np.log(self.obsInfo['energy_min'])
        logstep = np.log(self.obsInfo['energy_max']/self.obsInfo['energy_min'])/self.obsInfo['npix_e']
        pgrid  = "1:1024:#1024,1:1024:#1024"
        xygrid = '{0:.4f}:{1:.4f}:#{2:d},{3:.4f}:{4:.4f}:#{5:d}'.format(
                   self.obsInfo['x_min'], self.obsInfo['x_max'], self.obsInfo['npix_s'],
                   self.obsInfo['y_min'], self.obsInfo['y_max'], self.obsInfo['npix_s'])

        dict_exposure_maps = {}
        e_min = []
        e_max = []
        for i in range(0, self.obsInfo['npix_e']):

            # calculate the instrument map
            ############################
            # the instrument map is essentially the product of the mirror effective area
            # projected onto the detector surface with
            # the detector quantum efficiency, [units = cm**(2) counts/photon], and also accounts for bad pixels
            # see https://cxc.harvard.edu/ciao/ahelp/mkinstmap.html

            rt.mkinstmap.punlearn()
            instmap_dic = {}
            expmap_dic = {}

            if self.obsInfo['energy_ranges']:
                src_e_min = self.obsInfo['energy_ranges'][i]
                src_e_max = self.obsInfo['energy_ranges'][i+1]
            else:
                src_e_min = np.round(np.exp( logemin + i*logstep ), decimals=10)
                src_e_max = np.round(np.exp( logemin + (i+1.)*logstep ),   decimals=10)
            e_min.append(src_e_min)
            e_max.append(src_e_max)
            # if the number of energy sub-bins is one pick the center of the channel
            if self.obsInfo['exp_ebins_per_bin'] == 1:
                energy = 0.5*(src_e_max + src_e_min)
                specfile = 'NONE'
            # for more than one energy sub-bin, request a flat spectrum over the channel
            else:
                bins = np.linspace(src_e_min, src_e_max, self.obsInfo['exp_ebins_per_bin'], endpoint=True)
                specfile = outroot + 'spec-temp.dat'
                with open(specfile, "w+") as sf:
                    for energy in bins:
                        sf.write('{:.9f}\t{:9f}\n'.format(energy, 1./self.obsInfo['exp_ebins_per_bin']))
                energy = 1.0

            # compute the instrument map for each channel
            # note: the monoenergy keyword is only used if spectrumfile is None
            for det in det_num:
                subdet = 'ACIS-{:d}' .format(det)
                outf   = outroot + '_imap{:d}-{:.9f}.instmap'.format(det, src_e_min)
                instmap_dic[det] = outf
                rt.mkinstmap(outfile=outf, monoenergy=energy, pixelgrid=pgrid,
                             obsfile=self.obsInfo['event_file'], detsubsys=subdet,
                             maskfile=self.obsInfo['mask_file'], clobber="yes", spectrumfile=specfile)


            # individual detector's exposure maps
            #####################################
            # the exposure map combines the instrument map with the aspect solution and can be used to
            # convert counts to flux with normalize set to 'no' the units are
            # (time) * (effective area) [sec * cm**(2) counts/photon]
            # see https://cxc.harvard.edu/ciao/ahelp/mkexpmap.html
            #TODO what about not combining data and exposure? different psf? etc?
            rt.mkexpmap.punlearn()

            for det in det_num:
                outf = outroot  + '_expmap-{:d}-{:.9f}.expmap' .format(det, src_e_min)
                expmap_dic[det] = outf
                rt.mkexpmap(instmapfile=instmap_dic[det], outfile=outf, asphistfile=asphist_dic[det],
                            xygrid=xygrid, clobber='yes', normalize='no')

            # combined exposure maps for all detectors
            ##########################################
            # see https://cxc.cfa.harvard.edu/ciao/ahelp/dmimgcalc.html
            rt.dmimgcalc.punlearn()

            ifile = [expmap_dic[det] for det in expmap_dic.keys()]
            opstr = 'imgout=img1'
            for d in range(2,len(expmap_dic)+1):
                opstr = opstr + '+img{:d}'.format(d)
            outfs  = outroot + '_{:.9f}.expmap'.format(src_e_min)

            rt.dmimgcalc(infile=ifile, infile2='none', out=outfs, operation=opstr, clobber='yes')
            dict_exposure_maps[i] = outfs

            # clean intermediate files
            to_remove  = [instmap_dic[kk] for kk in instmap_dic.keys()]
            to_remove += [expmap_dic[kk] for kk in expmap_dic.keys()]
            to_remove += [specfile]
            for file in to_remove:
                os.remove(file)

        # read in the maps and initialize the exposure-filed
        ####################################################
        expmap = np.zeros([self.obsInfo['npix_s'], self.obsInfo['npix_s'], self.obsInfo['npix_e']])

        for i in range(0, self.obsInfo['npix_e']):
            with fits.open(dict_exposure_maps[i]) as mapfile:
                expmap[:,:,i] = mapfile['PRIMARY'].data

        # remove all energy-independent files
        to_remove  = [dict_exposure_maps[kk] for kk in dict_exposure_maps.keys()]
        to_remove += [asphist_dic[kk] for kk in asphist_dic.keys()]
        for file in to_remove:
            os.remove(file)
        print(f'Generated exposure for energy ranges {set(e_min + e_max)}')
        message_exposure(self.obsInfo)
        return expmap


    def get_psf_fromsim(self, location, outroot, num_rays=1e4, detector_type=None, aspect_blur=None):

        """
        Obtain the Point Spread Function (PSF) from simulations at the specified position in all energy channels.

        This method simulates the PSF at a given celestial location using MARX simulations. 
        The PSF is computed for the center of each energy channel.

        Parameters:
        -----------
        location : tuple
            Location at which to compute the PSF in celestial coordinates (RA, DEC) in units of degrees.
        outroot : str
            Directory where the intermediate MARX files are saved.
        num_rays : int, optional
            Number of detected rays in the simulation. Default is 1e4.
        detector_type : str, optional
            Type of detector used, either 'ACIS-I' or 'ACIS-S'. If None, the detector type from obsInfo will be used.
            Default is None.
        aspect_blur : float, optional
            Accounts for the observed widening of the PSF with respect to simulations. If None, values suggested by the
            CXC team will be used. Default is None.

        Returns:
        --------
        np.array
            A 3D numpy array (npix_e x npix_s x npix_s) with the simulated PSF.
        """

        self.psf_sim_coords.append(location)

        # 1. get detector parameters
        ############################
        if detector_type == None:
            detector_type = self.obsInfo['instrument']
            # 2.a) detector type
        if detector_type=='ACIS-I':
            det_short = 'AI2'
        elif detector_type=='ACIS-S':
            det_short = 'AS1'
        else:
            print('Invalid detector choice: {:s}'.format(detector_type))
            exit()

        # 2.b) aspect blur
        # https://cxc.cfa.harvard.edu/ciao/why/aspectblur.html (observational foundation)
        # https://cxc.cfa.harvard.edu/ciao/PSFs/chart2/caveats.html (suggested values)
        if aspect_blur==None and det_short=='AI2':
            aspect_blur = 0.2
        if aspect_blur==None and det_short=='AS1':
            aspect_blur = 0.25

        # 2.c) detector offset
        marx_file = os.environ['MARX_DATA_DIR'] + '/caldb/telD1999-07-23aimptsN0002.fits'\
                    + '[AIMPOINTS][AIMPOINT_NAME=' + det_short + '][cols AIMPOINT]'

        marx_nom = rt.dmlist(infile=marx_file, opt='data').splitlines()[7]
        marx_nom = re.findall(r"[-+]?\d*\.\d+|\d+", marx_nom)

        detoffset_x = float(rt.dmkeypar(infile=self.obsInfo['event_file'],keyword='SIM_X',echo=True))\
                      - float(marx_nom[1])
        detoffset_z = float(rt.dmkeypar(infile=self.obsInfo['event_file'],keyword='SIM_Z',echo=True))\
                      - float(marx_nom[3])



        # 2. get observation parameters
        ###############################

        # 2.b) Nominal Pointing
        pointing_ra   = rt.dmkeypar(infile=self.obsInfo['event_file'], keyword='RA_NOM',  echo=True)
        pointing_dec  = rt.dmkeypar(infile=self.obsInfo['event_file'], keyword='DEC_NOM', echo=True)
        pointing_roll = rt.dmkeypar(infile=self.obsInfo['event_file'], keyword='ROLL_NOM',echo=True)

        # 2.c) Timing Information
        tstart   = rt.dmkeypar(infile=self.obsInfo['event_file'],keyword='TSTART',  echo=True)


        # 3. set marx parameters
        ########################
        # see https://space.mit.edu/cxc/marx/inbrief/simsetup.html for details
        # negative NumRays specifies the detected number of rays not the generated number (some will scatter and not
        # reach the detector)
        # DetIdeal suppresses the detector quantum efficiency which is already accounted for by the exposure map

        marxpara_file = outroot + '_marx.par'
        marxpara_orig = os.environ['MARX_ROOT'] + '/share/marx/pfiles/marx.par'
        os.system('cp ' + marxpara_orig + ' ' + marxpara_file)

        marxpara_dict = {
            "SourceRA" : '{:.9f}'.format(location[0]),
            "SourceDEC" : '{:.9f}'.format(location[1]),
            "SourceType" : "POINT",
            "SpectrumType" : "FLAT",
            "SourceFlux" : '{:.9f}'.format(1.e-3),
            "ExposureTime" : "{:.1f}" .format(0.0),
            "NumRays" : "{:d}".format(+1*np.abs(num_rays).astype(int)),
            "RA_Nom" : pointing_ra,
            "Dec_Nom" : pointing_dec,
            "Roll_Nom" : pointing_roll,
            "DetOffsetX" : "{:.9f}" .format(detoffset_x),
            "DetOffsetZ" : "{:.9f}" .format(detoffset_z),
            "AspectBlur" : "{:.5f}" .format(aspect_blur),
            "DetectorType" : detector_type,
            "DetIdeal" : "yes",
            "GratingType" : "NONE",
            "MirrorType" : "HRMA",
            "HRMA_Ideal" : "no",
            "HRMAVig" : "1.0",
            "DitherModel" : "INTERNAL", #TODO without and later for whole image?
            "TStart" : tstart,
            "Verbose" : "no",
            "ACIS_Frame_Transfer_Time" : "0.000",
            "HRMA_Use_Struts" : "yes", # FIXME Find out of if "yes" or "no"
            "DetExtendFlag" : "yes"
        }

        # 4. run marx simulations for each energy bin
        #############################################
        logemin = np.log(self.obsInfo['energy_min'])
        logstep = np.log(self.obsInfo['energy_max']/self.obsInfo['energy_min'])/self.obsInfo['npix_e']
        sim_dic = {}

#        shift = self.obsInfo['dat_length_xy']/self.obsInfo['dat_nbin_loc'] # = shift by half a pixel
#        psf_xmin = src_x - self.obsInfo['dat_length_xy'] - shift
#        psf_xmax = src_x + self.obsInfo['dat_length_xy'] - shift
#        psf_ymin = src_y - self.obsInfo['dat_length_xy'] - shift
#        psf_ymax = src_y + self.obsInfo['dat_length_xy'] - shift

        print('Generating PSFs for individual energy bins...\n')

        for i in range(0, self.obsInfo['npix_e']):

            src_e_min = np.exp( logemin + i*logstep )
            src_e_max = np.exp( logemin + (i+1.)*logstep )
            outdir  = outroot + "_e{:d}.dir".format(i)
            outfits = outroot + "_e{:d}.fits".format(i)

            marxpara_dict["MinEnergy"] = "{:.9f}".format(src_e_min)
            marxpara_dict["MaxEnergy"] = "{:.9f}".format(src_e_max)
            marxpara_dict["OutputDir"] = outdir
            pset(marxpara_file, marxpara_dict)

            subprocess.call(["marx", "@@" + marxpara_file])
            subprocess.call(["marx2fits", "--pixadj=EDSER", outdir, outfits])
            #TODO what about pixadj =EXACT
            #https://cxc.cfa.harvard.edu/ciao/threads/marx/index.html#opps

            # 5. transform the eventfile to an image
            #######################################

            # 5.a) filter events for the FOV
            outfits2 = outroot + 'psf_e{:d}.fits'.format(i)
            infile = outfits + '[EVENTS][x={0:.1f}:{1:.1f}, y={2:.1f}:{3:.1f}]'.format(self.obsInfo['x_min'],
                                                                                       self.obsInfo['x_max'],
                                                                                       self.obsInfo['y_min'],
                                                                                       self.obsInfo['y_max'])
            infile += '[bin x={:.1f}:{:.1f}:#{:d}, y={:.1f}:{:.1f}:#{:d}]'.format(self.obsInfo['x_min'],
                                                                                  self.obsInfo['x_max'],
                                                                                  self.obsInfo['npix_s'],
                                                                                  self.obsInfo['y_min'],
                                                                                  self.obsInfo['y_max'],
                                                                                  self.obsInfo['npix_s'])
            infile += '[opt type=i4]'
            sim_dic[i] = outfits2
            rt.dmcopy(infile=infile, outfile=outfits2, clobber='yes')

            # clean the intermediate products
            shutil.rmtree(outdir)
            os.remove(outfits)

        # 5.b) write events to an array
        psf = np.zeros([self.obsInfo['npix_s'], self.obsInfo['npix_s'], self.obsInfo['npix_e']])
        for i in range(0, self.obsInfo['npix_e']):
            with fits.open(sim_dic[i]) as psffile:
                dat = psffile['PRIMARY'].data
                psf[:,:,i] = dat
        psf = psf.astype(int)

        # clean psf fits images
        for i in range(self.obsInfo['npix_e']):
                os.remove(sim_dic[i])
        os.remove(marxpara_file)

        return(psf)
