import ast

class HitObject:
    def __init__(self,x=0,y=0,time=0,type=0,csstype="None",hitSound=0,addition=[0,0,0,0,""],
                 slidertype="",curvePoints=[],repeat=0,pixelLength=0.0,
                 edgeHitsounds=[],edgeAdditions=[],endTime=0):
        self.x = x
        self.y = y
        self.time = time
        self.type = type
        self.csstype = csstype
        self.hitSound = hitSound
        self.addition = addition
        self.slidertype = slidertype
        self.curvePoints = curvePoints
        self.repeat = repeat
        self.pixelLength = pixelLength
        self.edgeHitsounds = edgeHitsounds
        self.edgeAdditions = edgeAdditions
        self.endTime = endTime

class Beatmap:
    def __init__(self):
        self.AudioFilename = ""
        self.AudioLeadIn = 0
        self.PreviewTime = 0
        self.Countdown = False
        self.SampleSet = ""
        self.StackLeniency = 0.0
        self.Mode = 0
        self.LetterboxInBreaks = False
        self.EpilepsyWarning = False
        self.WidescreenStoryboard = False

        self.Bookmarks = []
        self.DistanceSpacing = 0.0
        self.BeatDivisor = 0
        self.GridSize = 0
        self.TimelineZoom = 0

        self.Title = ""
        self.TitleUnicode = ""
        self.Artist = ""
        self.ArtistUnicode = ""
        self.Creator = ""
        self.Version = ""
        self.Source = ""
        self.Tags = []
        self.BeatmapID = 0
        self.BeatmapSetID = 0
        
        self.HPDrainRate = 0.0
        self.CircleSize = 0.0
        self.OverallDifficulty = 0.0
        self.ApproachRate = 0.0
        self.SliderMultiplier = 1.0
        self.SliderTickRate = 0.0
        
        self.Events = []
        
        self.TimingPoints = []
        
        self.ComboColours = {}
        
        self.HitObjects = []

    def calculate_slider_durations(self):
        tbb = 300
        itv = -100
        tpindex = 0
        for ho in self.HitObjects:
            if ho.csstype == "slider":
                while tpindex < len(self.TimingPoints) and self.TimingPoints[tpindex][0] <= ho.time:
                    if self.TimingPoints[tpindex][6]:
                        tbb = self.TimingPoints[tpindex][1]
                    else:
                        itv = self.TimingPoints[tpindex][1]
                    tpindex += 1
                ho.endTime = int((tbb * itv * ho.pixelLength)//(-10000 * self.SliderMultiplier))
        

def bitfield(n):
    return [1 if digit=='1' else 0 for digit in bin(n)[2:]][::-1]

def convertType(string):
    try:
        return ast.literal_eval(string)

    except ValueError:
        return string
    except SyntaxError:
        return string

def readLine(line):
    split = line.split(":")
    return [s.strip() for s in split]

def convertSplit(string, split):
    return [convertType(k) for k in string.split(split)]

def readBeatmap(file_location):
    bm = Beatmap()
    section = ""
    with open (file_location, 'rt', encoding="utf8") as f:
        for l in f:
            ls = l.strip()
            split = readLine(ls)
            
            if len(ls) == 0:
                continue
            if ls[0] == "[":
                section = ls
                continue

            if section == "[General]":
                setattr(bm, split[0], convertType(split[1]))
                
            elif section == "[Editor]":
                if split[0] == "Bookmarks":
                    bm.Bookmarks = convertSplit(split[1], ",")
                    continue
                setattr(bm, split[0], convertType(split[1]))
                
            elif section == "[Metadata]":
                if split[0] == "Tags":
                    bm.Tags = split[1].split(" ")
                setattr(bm, split[0], convertType(split[1]))
                
            elif section == "[Difficulty]":
                setattr(bm, split[0], convertType(split[1]))
                
            elif section == "[Events]":
                bm.Events.append(ls)
                
            elif section == "[TimingPoints]":
                bm.TimingPoints.append(convertSplit(ls, ","))
                
            elif section == "[Colours]":
                bm.ComboColours[split[0]] = convertSplit(split[1], ",")
                
            elif section == "[HitObjects]":
                lss = convertSplit(ls, ",")
                typebits = bitfield(lss[3])
                
                if typebits[0] == 1:  # Circle
                    bm.HitObjects.append(HitObject(lss[0],lss[1],lss[2],lss[3],"circle",lss[4], convertSplit(lss[5], ":")))
                    
                elif typebits[1] == 1:  # Slider
                    bm.HitObjects.append(HitObject(lss[0],lss[1],lss[2],lss[3],"slider",lss[4],None,
                                                   lss[5].split("|")[0],[convertSplit(t, ":") for t in lss[5].split("|")[1:]],
                                                   lss[6],lss[7],None,None))
                    try:
                        bm.HitObjects[-1].edgeHitsounds = convertSplit(lss[8], "|")
                        bm.HitObjects[-1].edgeAdditions = [convertSplit(t, ":") for t in lss[9].split("|")]
                        bm.HitObjects[-1].addition = convertSplit(lss[10], ":")
                    except IndexError:
                        pass
                    
                elif typebits[3] == 1:  # Spinner
                    bm.HitObjects.append(HitObject(lss[0],lss[1],lss[2],lss[3],"spinner",lss[4],convertSplit(lss[6], ":"), endTime=lss[5]))
                    
    bm.calculate_slider_durations()
    return bm
