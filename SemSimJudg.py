#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Tue 28 Oct 2025 12:38:15 PM CDT
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'SemSimJudg'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'Subject_ID': 'MCI',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = False
_winSize = [1920, 1200]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'SemSimJudg_data/%s_%s_%s' % (expInfo['Subject_ID'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/home/testadmin/Documents/MCI_similarity_judgment/SemSimJudg.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('welcomeEndKey') is None:
        # initialise welcomeEndKey
        welcomeEndKey = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='welcomeEndKey',
        )
    if deviceManager.getDevice('Task_response') is None:
        # initialise Task_response
        Task_response = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='Task_response',
        )
    if deviceManager.getDevice('breakEndKey') is None:
        # initialise breakEndKey
        breakEndKey = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='breakEndKey',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # enter 'rush' mode (raise CPU priority)
    if not PILOTING:
        core.rush(True, realtime=True)
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Welcome" ---
    welcomeText = visual.TextStim(win=win, name='welcomeText',
        text="Welcome to the study\n\nPress 'K' for left\nPress 'L' for right\n\nPress SPACE to start the task\n",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    welcomeEndKey = keyboard.Keyboard(deviceName='welcomeEndKey')
    
    # --- Initialize components for Routine "Fixation" ---
    initFixText = visual.TextStim(win=win, name='initFixText',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Priming" ---
    FixCross = visual.TextStim(win=win, name='FixCross',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    Target_text = visual.TextStim(win=win, name='Target_text',
        text='',
        font='Arial',
        pos=(0, 0.07), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    LeftWord = visual.TextStim(win=win, name='LeftWord',
        text='',
        font='Arial',
        pos=(-0.13, -0.07), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    RightWord = visual.TextStim(win=win, name='RightWord',
        text='',
        font='Arial',
        pos=(0.13, -0.07), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    Task_response = keyboard.Keyboard(deviceName='Task_response')
    # Run 'Begin Experiment' code from RT_Acc_tracker
    # Initialize variables for feedback
    response_times = []
    accuracy = []
    
    # --- Initialize components for Routine "Break" ---
    # Run 'Begin Experiment' code from breakCheck
    breakRoutineStart = False
    
    # Run 'Begin Experiment' code from feedbackCode
    import statistics
    breakText = visual.TextStim(win=win, name='breakText',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    breakEndKey = keyboard.Keyboard(deviceName='breakEndKey')
    
    # --- Initialize components for Routine "Fixation2" ---
    breakFixText = visual.TextStim(win=win, name='breakFixText',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from breakFixCheck
    breakFixationStart = False
    
    # --- Initialize components for Routine "Thank_You" ---
    Thanks_text = visual.TextStim(win=win, name='Thanks_text',
        text='All done!\n\nThank you for completing the task',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Welcome" ---
    # create an object to store info about Routine Welcome
    Welcome = data.Routine(
        name='Welcome',
        components=[welcomeText, welcomeEndKey],
    )
    Welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for welcomeEndKey
    welcomeEndKey.keys = []
    welcomeEndKey.rt = []
    _welcomeEndKey_allKeys = []
    # store start times for Welcome
    Welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Welcome.tStart = globalClock.getTime(format='float')
    Welcome.status = STARTED
    thisExp.addData('Welcome.started', Welcome.tStart)
    Welcome.maxDuration = None
    # keep track of which components have finished
    WelcomeComponents = Welcome.components
    for thisComponent in Welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Welcome" ---
    Welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcomeText* updates
        
        # if welcomeText is starting this frame...
        if welcomeText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcomeText.frameNStart = frameN  # exact frame index
            welcomeText.tStart = t  # local t and not account for scr refresh
            welcomeText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcomeText, 'tStartRefresh')  # time at next scr refresh
            # update status
            welcomeText.status = STARTED
            welcomeText.setAutoDraw(True)
        
        # if welcomeText is active this frame...
        if welcomeText.status == STARTED:
            # update params
            pass
        
        # *welcomeEndKey* updates
        waitOnFlip = False
        
        # if welcomeEndKey is starting this frame...
        if welcomeEndKey.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcomeEndKey.frameNStart = frameN  # exact frame index
            welcomeEndKey.tStart = t  # local t and not account for scr refresh
            welcomeEndKey.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcomeEndKey, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcomeEndKey.started')
            # update status
            welcomeEndKey.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(welcomeEndKey.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(welcomeEndKey.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if welcomeEndKey.status == STARTED and not waitOnFlip:
            theseKeys = welcomeEndKey.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _welcomeEndKey_allKeys.extend(theseKeys)
            if len(_welcomeEndKey_allKeys):
                welcomeEndKey.keys = _welcomeEndKey_allKeys[-1].name  # just the last key pressed
                welcomeEndKey.rt = _welcomeEndKey_allKeys[-1].rt
                welcomeEndKey.duration = _welcomeEndKey_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Welcome" ---
    for thisComponent in Welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Welcome
    Welcome.tStop = globalClock.getTime(format='float')
    Welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Welcome.stopped', Welcome.tStop)
    # check responses
    if welcomeEndKey.keys in ['', [], None]:  # No response was made
        welcomeEndKey.keys = None
    thisExp.addData('welcomeEndKey.keys',welcomeEndKey.keys)
    if welcomeEndKey.keys != None:  # we had a response
        thisExp.addData('welcomeEndKey.rt', welcomeEndKey.rt)
        thisExp.addData('welcomeEndKey.duration', welcomeEndKey.duration)
    thisExp.nextEntry()
    # the Routine "Welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Fixation" ---
    # create an object to store info about Routine Fixation
    Fixation = data.Routine(
        name='Fixation',
        components=[initFixText],
    )
    Fixation.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Fixation
    Fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Fixation.tStart = globalClock.getTime(format='float')
    Fixation.status = STARTED
    thisExp.addData('Fixation.started', Fixation.tStart)
    Fixation.maxDuration = None
    # keep track of which components have finished
    FixationComponents = Fixation.components
    for thisComponent in Fixation.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Fixation" ---
    Fixation.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *initFixText* updates
        
        # if initFixText is starting this frame...
        if initFixText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            initFixText.frameNStart = frameN  # exact frame index
            initFixText.tStart = t  # local t and not account for scr refresh
            initFixText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(initFixText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'initFixText.started')
            # update status
            initFixText.status = STARTED
            initFixText.setAutoDraw(True)
        
        # if initFixText is active this frame...
        if initFixText.status == STARTED:
            # update params
            pass
        
        # if initFixText is stopping this frame...
        if initFixText.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > initFixText.tStartRefresh + 2.0-frameTolerance:
                # keep track of stop time/frame for later
                initFixText.tStop = t  # not accounting for scr refresh
                initFixText.tStopRefresh = tThisFlipGlobal  # on global time
                initFixText.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'initFixText.stopped')
                # update status
                initFixText.status = FINISHED
                initFixText.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Fixation.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Fixation.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Fixation" ---
    for thisComponent in Fixation.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Fixation
    Fixation.tStop = globalClock.getTime(format='float')
    Fixation.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Fixation.stopped', Fixation.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Fixation.maxDurationReached:
        routineTimer.addTime(-Fixation.maxDuration)
    elif Fixation.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('SemSimJudg_all_stim.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "Priming" ---
        # create an object to store info about Routine Priming
        Priming = data.Routine(
            name='Priming',
            components=[FixCross, Target_text, LeftWord, RightWord, Task_response],
        )
        Priming.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from breakTrigger
        # Schedule a break after this trial
        if (trials.thisN + 1) % 20 == 0:
            breakRoutineStart = True
            if trials.thisN + 1 != len(trials.trialList):
                breakFixationStart = True
        FixCross.setText('+')
        Target_text.setText(Target)
        LeftWord.setText(Left)
        RightWord.setText(Right)
        # create starting attributes for Task_response
        Task_response.keys = []
        Task_response.rt = []
        _Task_response_allKeys = []
        # store start times for Priming
        Priming.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Priming.tStart = globalClock.getTime(format='float')
        Priming.status = STARTED
        thisExp.addData('Priming.started', Priming.tStart)
        Priming.maxDuration = None
        # keep track of which components have finished
        PrimingComponents = Priming.components
        for thisComponent in Priming.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Priming" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        Priming.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *FixCross* updates
            
            # if FixCross is starting this frame...
            if FixCross.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                FixCross.frameNStart = frameN  # exact frame index
                FixCross.tStart = t  # local t and not account for scr refresh
                FixCross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixCross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'FixCross.started')
                # update status
                FixCross.status = STARTED
                FixCross.setAutoDraw(True)
            
            # if FixCross is active this frame...
            if FixCross.status == STARTED:
                # update params
                pass
            
            # if FixCross is stopping this frame...
            if FixCross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > FixCross.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    FixCross.tStop = t  # not accounting for scr refresh
                    FixCross.tStopRefresh = tThisFlipGlobal  # on global time
                    FixCross.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'FixCross.stopped')
                    # update status
                    FixCross.status = FINISHED
                    FixCross.setAutoDraw(False)
            
            # *Target_text* updates
            
            # if Target_text is starting this frame...
            if Target_text.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                Target_text.frameNStart = frameN  # exact frame index
                Target_text.tStart = t  # local t and not account for scr refresh
                Target_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Target_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Target_text.started')
                # update status
                Target_text.status = STARTED
                Target_text.setAutoDraw(True)
            
            # if Target_text is active this frame...
            if Target_text.status == STARTED:
                # update params
                pass
            
            # *LeftWord* updates
            
            # if LeftWord is starting this frame...
            if LeftWord.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                LeftWord.frameNStart = frameN  # exact frame index
                LeftWord.tStart = t  # local t and not account for scr refresh
                LeftWord.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(LeftWord, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'LeftWord.started')
                # update status
                LeftWord.status = STARTED
                LeftWord.setAutoDraw(True)
            
            # if LeftWord is active this frame...
            if LeftWord.status == STARTED:
                # update params
                pass
            
            # *RightWord* updates
            
            # if RightWord is starting this frame...
            if RightWord.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                RightWord.frameNStart = frameN  # exact frame index
                RightWord.tStart = t  # local t and not account for scr refresh
                RightWord.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(RightWord, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'RightWord.started')
                # update status
                RightWord.status = STARTED
                RightWord.setAutoDraw(True)
            
            # if RightWord is active this frame...
            if RightWord.status == STARTED:
                # update params
                pass
            
            # *Task_response* updates
            waitOnFlip = False
            
            # if Task_response is starting this frame...
            if Task_response.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                Task_response.frameNStart = frameN  # exact frame index
                Task_response.tStart = t  # local t and not account for scr refresh
                Task_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Task_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Task_response.started')
                # update status
                Task_response.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(Task_response.clock.reset)  # t=0 on next screen flip
            if Task_response.status == STARTED and not waitOnFlip:
                theseKeys = Task_response.getKeys(keyList=['a','s','k','l'], ignoreKeys=["escape"], waitRelease=False)
                _Task_response_allKeys.extend(theseKeys)
                if len(_Task_response_allKeys):
                    Task_response.keys = _Task_response_allKeys[0].name  # just the first key pressed
                    Task_response.rt = _Task_response_allKeys[0].rt
                    Task_response.duration = _Task_response_allKeys[0].duration
                    # was this correct?
                    if (Task_response.keys == str(CorrectResp_RH)) or (Task_response.keys == CorrectResp_RH):
                        Task_response.corr = 1
                    else:
                        Task_response.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Priming.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Priming.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Priming" ---
        for thisComponent in Priming.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Priming
        Priming.tStop = globalClock.getTime(format='float')
        Priming.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Priming.stopped', Priming.tStop)
        # check responses
        if Task_response.keys in ['', [], None]:  # No response was made
            Task_response.keys = None
            # was no response the correct answer?!
            if str(CorrectResp_RH).lower() == 'none':
               Task_response.corr = 1;  # correct non-response
            else:
               Task_response.corr = 0;  # failed to respond (incorrectly)
        # store data for trials (TrialHandler)
        trials.addData('Task_response.keys',Task_response.keys)
        trials.addData('Task_response.corr', Task_response.corr)
        if Task_response.keys != None:  # we had a response
            trials.addData('Task_response.rt', Task_response.rt)
            trials.addData('Task_response.duration', Task_response.duration)
        # Run 'End Routine' code from RT_Acc_tracker
        # Append RT and Acc to lists for feedback
        accuracy.append(Task_response.corr)
        if Task_response.corr == 1:
            response_times.append(Task_response.rt)
        # the Routine "Priming" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Break" ---
        # create an object to store info about Routine Break
        Break = data.Routine(
            name='Break',
            components=[breakText, breakEndKey],
        )
        Break.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from breakCheck
        if not breakRoutineStart:
            continueRoutine = False  # Skip the break if it's not time
        
        # Run 'Begin Routine' code from feedbackCode
        feedbackMsg = 'fb_msg_init'
        if breakRoutineStart:
            # Compute averages for the previous block
            if accuracy == []:
                feedbackMsg = "End of this block \n \n NO RESPONSES RECORDED!"
            elif response_times == []:
                mean_acc = statistics.mean(accuracy)
                feedbackMsg = "End of this block \n \n" + "No correct responses\n" + "Press SPACE to continue"      
            else:
                mean_rt = statistics.mean(response_times)
                mean_acc = statistics.mean(accuracy)
                feedbackMsg = "End of this block \n \n" + "Average response time: " + str(round(mean_rt, 2)) + " seconds \n" + "Press SPACE to continue"
        breakText.setText(feedbackMsg)
        # create starting attributes for breakEndKey
        breakEndKey.keys = []
        breakEndKey.rt = []
        _breakEndKey_allKeys = []
        # store start times for Break
        Break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Break.tStart = globalClock.getTime(format='float')
        Break.status = STARTED
        thisExp.addData('Break.started', Break.tStart)
        Break.maxDuration = None
        # keep track of which components have finished
        BreakComponents = Break.components
        for thisComponent in Break.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Break" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        Break.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *breakText* updates
            
            # if breakText is starting this frame...
            if breakText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                breakText.frameNStart = frameN  # exact frame index
                breakText.tStart = t  # local t and not account for scr refresh
                breakText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(breakText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'breakText.started')
                # update status
                breakText.status = STARTED
                breakText.setAutoDraw(True)
            
            # if breakText is active this frame...
            if breakText.status == STARTED:
                # update params
                pass
            
            # *breakEndKey* updates
            waitOnFlip = False
            
            # if breakEndKey is starting this frame...
            if breakEndKey.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                breakEndKey.frameNStart = frameN  # exact frame index
                breakEndKey.tStart = t  # local t and not account for scr refresh
                breakEndKey.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(breakEndKey, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'breakEndKey.started')
                # update status
                breakEndKey.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(breakEndKey.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(breakEndKey.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if breakEndKey.status == STARTED and not waitOnFlip:
                theseKeys = breakEndKey.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _breakEndKey_allKeys.extend(theseKeys)
                if len(_breakEndKey_allKeys):
                    breakEndKey.keys = _breakEndKey_allKeys[-1].name  # just the last key pressed
                    breakEndKey.rt = _breakEndKey_allKeys[-1].rt
                    breakEndKey.duration = _breakEndKey_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Break.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Break.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Break" ---
        for thisComponent in Break.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Break
        Break.tStop = globalClock.getTime(format='float')
        Break.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Break.stopped', Break.tStop)
        # Run 'End Routine' code from breakCheck
        breakRoutineStart = False
        
        # check responses
        if breakEndKey.keys in ['', [], None]:  # No response was made
            breakEndKey.keys = None
        trials.addData('breakEndKey.keys',breakEndKey.keys)
        if breakEndKey.keys != None:  # we had a response
            trials.addData('breakEndKey.rt', breakEndKey.rt)
            trials.addData('breakEndKey.duration', breakEndKey.duration)
        # the Routine "Break" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Fixation2" ---
        # create an object to store info about Routine Fixation2
        Fixation2 = data.Routine(
            name='Fixation2',
            components=[breakFixText],
        )
        Fixation2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from breakFixCheck
        if not breakFixationStart:
            continueRoutine = False  # Skip the fixation if it's not time
        else:
            # Reset lists for next block
            response_times = []
            accuracy = []
        
        # store start times for Fixation2
        Fixation2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Fixation2.tStart = globalClock.getTime(format='float')
        Fixation2.status = STARTED
        thisExp.addData('Fixation2.started', Fixation2.tStart)
        Fixation2.maxDuration = None
        # keep track of which components have finished
        Fixation2Components = Fixation2.components
        for thisComponent in Fixation2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Fixation2" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        Fixation2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *breakFixText* updates
            
            # if breakFixText is starting this frame...
            if breakFixText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                breakFixText.frameNStart = frameN  # exact frame index
                breakFixText.tStart = t  # local t and not account for scr refresh
                breakFixText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(breakFixText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'breakFixText.started')
                # update status
                breakFixText.status = STARTED
                breakFixText.setAutoDraw(True)
            
            # if breakFixText is active this frame...
            if breakFixText.status == STARTED:
                # update params
                pass
            
            # if breakFixText is stopping this frame...
            if breakFixText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > breakFixText.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    breakFixText.tStop = t  # not accounting for scr refresh
                    breakFixText.tStopRefresh = tThisFlipGlobal  # on global time
                    breakFixText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'breakFixText.stopped')
                    # update status
                    breakFixText.status = FINISHED
                    breakFixText.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Fixation2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Fixation2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Fixation2" ---
        for thisComponent in Fixation2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Fixation2
        Fixation2.tStop = globalClock.getTime(format='float')
        Fixation2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Fixation2.stopped', Fixation2.tStop)
        # Run 'End Routine' code from breakFixCheck
        breakFixationStart = False
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Fixation2.maxDurationReached:
            routineTimer.addTime(-Fixation2.maxDuration)
        elif Fixation2.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if trials.trialList in ([], [None], None):
        params = []
    else:
        params = trials.trialList[0].keys()
    # save data for this loop
    trials.saveAsExcel(filename + '.xlsx', sheetName='trials',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    trials.saveAsText(filename + 'trials.csv', delim=',',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "Thank_You" ---
    # create an object to store info about Routine Thank_You
    Thank_You = data.Routine(
        name='Thank_You',
        components=[Thanks_text],
    )
    Thank_You.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Thank_You
    Thank_You.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Thank_You.tStart = globalClock.getTime(format='float')
    Thank_You.status = STARTED
    thisExp.addData('Thank_You.started', Thank_You.tStart)
    Thank_You.maxDuration = None
    # keep track of which components have finished
    Thank_YouComponents = Thank_You.components
    for thisComponent in Thank_You.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Thank_You" ---
    Thank_You.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Thanks_text* updates
        
        # if Thanks_text is starting this frame...
        if Thanks_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Thanks_text.frameNStart = frameN  # exact frame index
            Thanks_text.tStart = t  # local t and not account for scr refresh
            Thanks_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Thanks_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Thanks_text.started')
            # update status
            Thanks_text.status = STARTED
            Thanks_text.setAutoDraw(True)
        
        # if Thanks_text is active this frame...
        if Thanks_text.status == STARTED:
            # update params
            pass
        
        # if Thanks_text is stopping this frame...
        if Thanks_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Thanks_text.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                Thanks_text.tStop = t  # not accounting for scr refresh
                Thanks_text.tStopRefresh = tThisFlipGlobal  # on global time
                Thanks_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Thanks_text.stopped')
                # update status
                Thanks_text.status = FINISHED
                Thanks_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Thank_You.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Thank_You.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Thank_You" ---
    for thisComponent in Thank_You.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Thank_You
    Thank_You.tStop = globalClock.getTime(format='float')
    Thank_You.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Thank_You.stopped', Thank_You.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Thank_You.maxDurationReached:
        routineTimer.addTime(-Thank_You.maxDuration)
    elif Thank_You.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)
    # end 'rush' mode
    core.rush(False, realtime=False)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
