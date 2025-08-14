import os, sys, time, datetime, subprocess, json, shutil, copy, re, argparse, math
import numpy as np


# We are here
source=os.getcwd()
# Test if mandatory CREST output file is present
# Might be changed in the future to get rid of universal script dependency
if not os.path.isfile( source+"/crest_conformers.xyz" ):
   print( "\n Error: crest_conformers.xyz file is missing\n" )
   quit()


# Parse arguments
#-------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument( "--chrg",
                     dest="charge",
                     type=str,
                     default="0"
                   )
parser.add_argument( "--multi",
                     dest="multi",
                     type=str,
                     default="1"
                   )
parser.add_argument( "--env",
                     dest="environment",
                     choices=[
                               "acetone",
                               "acetonitrile",
                               "chloroform", "chcl3",
                               "dmso",
                               "gp",
                               "methanol",
                               "water"
                             ],
                   )
parser.add_argument( "--step",
                     dest="step",
                     choices=[
                               "1",
                               "2",
                               "3",
                               "sort",
                               "filter",
                               "shifts",
                               "ssccs",
                               "eval",
                               "nmr",
                               "all"
                             ],
                     required = True
                   )
parser.add_argument( "--thrG",
                     dest="thrG",
                     type=float,
                     default="20.0"
                   )
parser.add_argument( "--thrBM",
                     dest="thrBM",
                     type=float,
                     default="100.0"
                   )
parser.add_argument( "--freq",
                     dest="freq",
                     nargs=1,
                     type=float
                   )
parser.add_argument( "--XH",
                     dest="acidic_H",
                     nargs='+',
                     type=str
                   )
parser.add_argument( "--restart_f3",
                     dest="res_cyc",
                     nargs=1,
                     type=int,
                     default=None
                   )
args = parser.parse_args()


sscc_components = "{ SSFC }" # SSDSO, SSPSO, SSSD, SSALL
# another one to consider at some point: SpinSpinRThresh 10.0

if args.charge:
   charge=args.charge

if args.multi:
   multi=args.multi

fstep = args.step
thresh_G = float(args.thrG)
thresh_BM = float(args.thrBM)

XTB_solv_model="alpb"
if args.environment:
   solvent_env = args.environment
else: # try to determine solvent from filter_results.json, if present
   try:
      with open( source+"/filter_results.json", "r" ) as file:
         filter_results=json.load( file )
         if filter_results["Solvent"] is not None:
            solvent_env = filter_results["Solvent"]
         else:
            print("\n Error: environment not set\n")
            quit()
   except:
      print("\n Error: environment not set\n")
      quit()
 
if solvent_env == "chloroform" or solvent_env == "chcl3":
   solvent_ORCA = "chloroform"
   solvent_XTB  = "chcl3"
# special case methanol, because the ALPB model of XTB does not support it -> use GBSA instead
elif solvent_env == "methanol":
   XTB_solv_model="gbsa"
   solvent_ORCA = solvent_env
   solvent_XTB  = solvent_env
elif solvent_env == "gp":
   solvent_ORCA = "gp"
   solvent_XTB  = ""
else:
   solvent_ORCA = solvent_env
   solvent_XTB  = solvent_env

if args.freq:
   nmr_freq = args.freq[0]
else:
   nmr_freq = None

if args.acidic_H:
   acidic_H = args.acidic_H
else:
   acidic_H = None

if args.res_cyc is None:
   restart_f3_cycle = None
else:
   restart_f3_cycle = int(args.res_cyc[0])
#-------------------------------------------------------------------------------------------------------




# Provide reference shieldings for different DFT methods and CPCM solvent environments
#-------------------------------------------------------------------------------------------------------
References = { "KT3/pcSseg-3" : {
                                "acetone"      : {
                                                 "1H"  :  31.4388333,
                                                 "13C" : 185.6372500
                                                 },
                                "acetonitrile" : {
                                                 "1H"  :  31.4378333,
                                                 "13C" : 185.6577500
                                                 },
                                "chloroform"   : {
                                                 "1H"  :  31.4465000,
                                                 "13C" : 185.4872500
                                                 },
                                "dmso"         : {
                                                 "1H"  :  31.4375833,
                                                 "13C" : 185.6637500
                                                 },
                                "methanol"     : {
                                                 "1H"  :  31.4380000,
                                                 "13C" : 185.6542500
                                                 },
                                "water"        : {
                                                 "1H"  :  31.4370833,
                                                 "13C" : 185.6722500
                                                 },
                                "gp"           : {
                                                 "1H"  :  31.4370833,
                                                 "13C" : 185.6722500
                                                 }
                                }
             }
#-------------------------------------------------------------------------------------------------------




# Define constants for unit conversion
#-------------------------------------------------------------------------------------------------------
ha_to_kcal=627.5094740631
kcal_to_kj=4.184
#-------------------------------------------------------------------------------------------------------




# Set relative energy thresholds for filtering steps (kJ/mol)
#-------------------------------------------------------------------------------------------------------
filter_1_thr=20.0
filter_2_thr=15.0
filter_3_thr=10.0
#-------------------------------------------------------------------------------------------------------




func_sp="r2SCAN-3c"
#func_sp="tpss def2-TZVP def2/J RIJCOSX d3bj"
func_opt="r2SCAN-3c"
#func_opt="tpss def2-TZVP def2/J RIJCOSX d3bj"

func_shifts="KT3"
#func_shifts="pbe0"

#func_couplings="KT3"
func_couplings="pbe0"




def batch_settings( filter_step ):
#-------------------------------------------------------------------------------------------------------

   email="user@inet"

   if filter_step == "filter_1":
      opt="false"
      xtb="true"
      xtb_hess="false"
      nnodes = 1
      ntasks = 1
      mem    = 250
      time   = "12:00:00"
      parti  = "main"
      qos    = "standard"

   elif filter_step == "filter_2":
      opt="false"
      xtb="true"
      xtb_hess="true"
      nnodes = 1
      ntasks = 1
      mem    = 300
      time   = "12:00:00"
      parti  = "main"
      qos    = "standard"

   elif filter_step == "filter_3":
      opt="true"
      xtb="true"
      xtb_hess="true"
      nnodes = 1
      ntasks = 1
      mem    = 350
      time   = "3-00:00:00"
      parti  = "main"
      qos    = "standard"

   elif filter_step == "nmr_shifts":
      opt="false"
      xtb="false"
      xtb_hess="false"
      nnodes = 1
      ntasks = 16
      mem    = 2500
      time   = "14-00:00:00"
      parti  = "main"
      qos    = "standard"

   elif filter_step == "nmr_couplings":
      opt="false"
      xtb="false"
      xtb_hess="false"
      nnodes = 1
      ntasks = 16
      mem    = 1400
      time   = "14-00:00:00"
      parti  = "main"
      qos    = "standard"

   prog="ORCA"

   xtb_home="export XTBHOME=/home/rmueller/prog/xtb/6.5.1\n"       \
            "export XTBPATH=/home/rmueller/prog/xtb/6.5.1:$PATH\n" \
            "export PATH=$XTBHOME:$PATH\n"                         \
            "export PATH=$XTBHOME/bin:$PATH\n"                     \

   return nnodes, ntasks, mem, time, parti, qos, prog, xtb_home, email, opt, xtb, xtb_hess
#-------------------------------------------------------------------------------------------------------





# Universal SLURM batchfile template
# This template is automatically adjusted according to the needs of the actual process step
# In particular
#    NODES      : number of nodes
#    TASKS      : number of tasks
#    MEM        : requested memory
#    TIME       : requested queue time
#    PARTI      : partition of the ZEDAT Curta cluster
#    QOS        : quality of service of the ZEDAT Curta cluster
#    MAIL       : uer email address
#    PROG       : program module for ORCA on the ZEDAT Curta cluster
#    XTB        : XTB program information (location, bin, ...)
#    MULTI      : molecular multiplicity (ORCA, XTB)
#    CHARGE     : molecular charge (ORCA, XTB)
#    SOLV_XTB   : solvent
#    XTB_SMODEL : ALPB/GBSA solvent model of XTB
#    OPT_BOOL   : enable/disable structure optimizations
#    XTB_BOOL   : enable/disable usage of XTB
#    XTB_HESS   : enable/disable the single point hesssian calculation of XTB
#    SOURCE     : info file for which conformers are calculated
#-------------------------------------------------------------------------------------------------------
batch_file="""#!/bin/bash --login
#SBATCH -o DEFAULT.%A_%a.out
#SBATCH -J DEFAULT

#SBATCH --nodes={NODES}
#SBATCH --ntasks={TASKS}
#SBATCH --ntasks-per-core=1
#SBATCH --hint=nomultithread
#SBATCH --mem-per-cpu={MEM}

#SBATCH --time={TIME}
#SBATCH -p {PARTI}
#SBATCH --qos={QOS}

#SBATCH --mail-type=FAIL
#SBATCH --mail-user={MAIL}

#SBATCH --array=RANGE

#SBATCH --exclude=c091,c093,c095

module load {PROG}

{XTB}

NAME=run
CHARGE="{CHARGE}"
MULTI="{MULTI}"
SOLVENT={SOLV_XTB}
SMODEL={XTB_SMODEL}
OPT={OPT_BOOL}
XTB={XTB_BOOL}
XTBH={XTB_HESS}


export SOURCEDIR=$(pwd)
SOURCE_FILE={SOURCE}
conformer=$( head -n ${{SLURM_ARRAY_TASK_ID}} ${{SOURCE_FILE}} | tail -n 1 )
if [[ "${{OPT}}" == "true" ]]
then
   COORD_FILE="run.xyz"
else
   COORD_FILE="${{conformer}}.xyz"
fi


if ! [ -d /localscratch/${{USER}} ]; then mkdir /localscratch/${{USER}}; fi
export TEMPDIR=/localscratch/${{USER}}/${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
mkdir -p ${{TEMPDIR}}


clean_up() {{
   cd ${{TEMPDIR}}/conformer-${{conformer}}
   rm -f *.densities *.gbw &> /dev/null
   if grep COPT ${{NAME}}.out &> /dev/null
   then
      sed -i "s/ COPT/ OPT/" ${{NAME}}.out
   fi
   find . -maxdepth 1 -mindepth 1 ! -name "*.tmp" ! -name "*.tmp.*" -exec cp {{}} ${{SOURCEDIR}}/conformer-${{conformer}} &> /dev/null \;
   cp -r XTB ${{SOURCEDIR}}/conformer-${{conformer}} &> /dev/null
   if ! grep "ORCA TERMINATED NORMALLY" ${{NAME}}.out &> /dev/null
   then
      touch ${{SOURCEDIR}}/conformer-${{conformer}}/ERROR
   fi
   cd ${{SOURCEDIR}}
   rm -rf ${{TEMPDIR}}
   while true
   do
      if mkdir lock.prog 2> /dev/null
      then
         echo "${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}" >> ${{SOURCEDIR}}/finished_ids
         rm -rf lock.prog
         break
      fi
      sleep 0.666s
   done
}}


trap clean_up EXIT INT TERM SIGTERM
#--------------------------------------------------------------------------------------------
cd ${{TEMPDIR}}
mkdir conformer-${{conformer}}
cp ${{SOURCEDIR}}/conformer-${{conformer}}/* ${{TEMPDIR}}/conformer-${{conformer}}
cd conformer-${{conformer}}
sed -i "s/maxcore.*/maxcore ${{SLURM_MEM_PER_CPU}}/" ${{NAME}}
sed -i "s/nprocs.*/nprocs ${{SLURM_NTASKS}}/" ${{NAME}}
sed -i "s/.*xyzfile.*/\* xyzfile ${{CHARGE}} ${{MULTI}} ${{COORD_FILE}}/" ${{NAME}}
orcarun=`which orca`
{{ time ${{orcarun}} ${{TEMPDIR}}/conformer-${{conformer}}/${{NAME}} &> ${{TEMPDIR}}/conformer-${{conformer}}/${{NAME}}.out; }}
#--------------------------------------------------------------------------------------------
if grep "ORCA TERMINATED NORMALLY" ${{NAME}}.out &> /dev/null
then
   if [[ "${{XTB}}" == "true" ]]
   then
      mkdir XTB
      cp ${{COORD_FILE}} XTB
      cd XTB
      if [[ "${{XTBH}}" == "true" ]] ; then XTBH="--bhess"                ; else unset NCI     ; fi
      if ! [ -z ${{SOLVENT}} ]       ; then SOLVENT="--${{SMODEL}} ${{SOLVENT}}" ; else unset SOLVENT ; fi
      {{ time xtb ${{COORD_FILE}} ${{SOLVENT}} ${{XTBH}} &> ${{TEMPDIR}}/conformer-${{conformer}}/XTB/XTB.out; }}
      cd ${{TEMPDIR}}
   fi
fi
#--------------------------------------------------------------------------------------------

"""
#-------------------------------------------------------------------------------------------------------




# ORCA input files for the filtering steps / NMR shielding and sscc calculations
#-------------------------------------------------------------------------------------------------------
filter_1_input="""! B97-D3 def2-SV(P) GCP(DFT/SV(P))
! def2/J RIJCOSX
! LooseSCF
! NOSOSCF

%maxcore MEM

%scf
maxiter 900
end

%pal
nprocs NPROCS
end

* xyzfile 0 1 COORDS
"""


if solvent_ORCA != "gp":
   solvent_ORCA_batch = "! CPCM("+str(solvent_ORCA)+")"
else:
   solvent_ORCA_batch = ""

filter_2_input="""! {FUNC}
! TightSCF
! DEFGRID2
! NOSOSCF
{SOLV_ORCA}

%maxcore MEM

%scf
maxiter 900
end

%pal
nprocs NPROCS
end

* xyzfile 0 1 COORDS
""".format( FUNC=str(func_sp), SOLV_ORCA=str(solvent_ORCA_batch) )


filter_3_input="""! {FUNC} OPT
! TightSCF
! DEFGRID2
! NOSOSCF
{SOLV_ORCA}

%maxcore MEM

%scf
maxiter 900
end

%geom
maxiter 8
end

%pal
nprocs NPROCS
end

* xyzfile 0 1 COORDS
""".format( FUNC=str(func_opt), SOLV_ORCA=str(solvent_ORCA_batch) )


if func_shifts == "KT3":
   nmr_shifts_details="""%method
method dft
functional gga_xc_KT3
end

! PCSSEG-3
! RIJCOSX AutoAux
! TightSCF
! DEFGRID2
! NOSOSCF"""
elif func_shifts == "pbe0":
   nmr_shifts_details="""! pbe0
! D4
! def2-TZVP
! def2/J RIJCOSX
! TightSCF
! DEFGRID2
! NOSOSCF"""


nmr_shifts_input="""{FUNC}
! NMR
{SOLV_ORCA}

%maxcore MEM

%scf
   maxiter 300
end

%pal
   nprocs NPROCS
end

* xyzfile 0 1 COORDS

%EPRNMR
   NUCLEI = ALL C {{SHIFT}}
   NUCLEI = ALL H {{SHIFT}}
   origin giao
   giao_2el giao_2el_same_as_scf
   giao_1el giao_1el_analytic
END
""".format( FUNC=str(nmr_shifts_details), SOLV_ORCA=str(solvent_ORCA_batch) )


if func_couplings == "KT3":
   nmr_couplings_details="""%method
method dft
functional gga_xc_KT3
end

! pcJ-1
! RIJCOSX AutoAux
! TightSCF
! DEFGRID2
! NOSOSCF"""
elif func_couplings == "pbe0":
   nmr_couplings_details="""! pbe0
! D4
! aug-pcJ-0
! RIJCOSX AutoAux
! TightSCF
! DEFGRID2
! NOSOSCF"""


nmr_couplings_input="""{FUNC}
! NMR
{SOLV_ORCA}

%maxcore MEM

%scf
   maxiter 300
end

%pal
   nprocs NPROCS
end

* xyzfile 0 1 COORDS

%EPRNMR
   NUCLEI = ALL C {SSCOMP}
   NUCLEI = ALL H {SSCOMP}
   SpinSpinRThresh 8.0
END
""".format( FUNC=str(nmr_couplings_details), SOLV_ORCA=str(solvent_ORCA_batch), SSCOMP=str(sscc_components) )
#-------------------------------------------------------------------------------------------------------




def get_nr_atoms_conformers( path_to_file, input_file="crest_conformers.xyz" ):
#-------------------------------------------------------------------------------
   with open(path_to_file+"/"+input_file, "r") as file:

      for line in file:
         line = line.strip()
         natoms = int(line)
         break
      file.seek(0)

      nconformers=0
      for line in file:
         if re.match( " *"+str(natoms)+"$", line ) or re.match( "^"+str(natoms)+"$", line ):
            nconformers += 1

   return natoms, nconformers
#-------------------------------------------------------------------------------




def extract_cart_coord( path_to_file, natoms, input_file="crest_conformers.xyz", conformers_list=None ):
#-------------------------------------------------------------------------------------------------------
   get_all = False
   if conformers_list is None:
      get_all = True
   with open( path_to_file+"/"+input_file, "r" ) as file:
      conformer = 1
      conformer_file = str(conformer)+".xyz"
      block = 1
      block_size = natoms+2
      coords = []
      for line in file:
         if block > block_size:
            if get_all or ( str(conformer) in conformers_list ):
               conformer_file = str(conformer)+".xyz"
               with open( conformer_file, "a" ) as file:
                  for entry in coords:
                     if type(entry) == list:
                        file.write("{0} {1} {2} {3}\n".format( entry[0], entry[1], entry[2], entry[3] ) )
                     else:
                        file.write(entry+"\n")
            coords = []
            block=1
            conformer+=1
            conformer_file=str(conformer)+".xyz"

         if block <= block_size:
            if block == 1:
               coords.append( line.strip() )
            elif block == 2:
               coords.append( " " )
            else:
               line = line.strip()
               line = line.split()
               coords.append( [ line[0], float(line[1]), float(line[2]), float(line[3]) ] )
            block+=1

      if get_all or ( str(conformer) in conformers_list ):
         with open( conformer_file, "a" ) as file:
            for entry in coords:
               if type(entry) == list:
                  file.write( "{0} {1} {2} {3}\n".format( entry[0], entry[1], entry[2], entry[3] ) )
               else:
                  file.write( entry+"\n" )

   return
#-------------------------------------------------------------------------------------------------------




def comp_driver( path, task, conformers_list ):
#-------------------------------------------------------------------------------------------------------


   if task == "filter_1":
      message="filter 1"
   elif task == "filter_2":
      message="filter 2"
   elif task == "filter_3":
      message="filter 3"
   elif task == "nmr_shifts":
      message="nmr shifts"
   elif task == "nmr_couplings":
      message="nmr couplings"

   restart = False

   while True:

      os.chdir( path+"/"+task )

      if restart:
         if os.path.isfile( path+"/"+task+"/ids" ):
            if os.path.isfile( path+"/"+task+"/prev_ids" ):
               with open( path+"/"+task+"/ids", "r" ) as file:
                  data = file.read()
               with open( path+"/"+task+"/prev_ids", "r" ) as file:
                  data2 = file.read()
               data = data + data2
               with open( path+"/"+task+"/prev_ids", "w" ) as file:
                  file.write( data )
            else:
               os.rename( 'ids', 'prev_ids' )
         restart = False

      subprocess.run( [ "rm -f batch_*"      ], shell=True, text=True )
      subprocess.run( [ "rm -f *_batch*"     ], shell=True, text=True )
      subprocess.run( [ "rm -f ids"          ], shell=True, text=True )
      subprocess.run( [ "rm -f finished_ids" ], shell=True, text=True )


      n_batch=1
      n_conf=0
      for conf in conformers_list:
         with open( "batch_array_"+str(n_batch), "a") as file:
            file.write( str(conf)+"\n" )
         n_conf+=1
         # On Curta only job arrays with a max size of 5000 are permitted
         if n_conf == int(5000):
            n_conf=0
            n_batch+=1
      # after the loop if target number was hit exactly, the n_batch count is increased by one although there are no confs left -> reduce by one
      if n_conf == 0:
         n_batch-=1



      subprocess.run( ["touch finished_ids" ], shell=True, text=True)

      total_tasks = 0
      for thread in range( 1, n_batch+1 ):

         tasks = 0
         with open( "batch_array_"+str(thread), "r" ) as file:
            for line in file:
               tasks += 1
         total_tasks = total_tasks + tasks

         batch=""
         with open( "slurm_template", "r" ) as file:
            for line in file:
               # has to be edited here because of the restart procedure for calculations that didn't finish successfully
               if "#SBATCH --array=RANGE" in line:
                  batch = batch+"#SBATCH --array=1-"+str(tasks)+"\n"
               elif "SOURCE_FILE=" in line:
                  batch = batch+"SOURCE_FILE=batch_array_"+str(thread)+"\n"
               else:
                  batch=batch+line

         with open( "batch_file_"+str(thread), "w" ) as file:
            file.write(batch)

         batch_id=subprocess.run([ "sbatch batch_file_"+str(thread) ], shell=True, capture_output=True, text=True)
         batch_id=batch_id.stdout.strip()
         batch_id=batch_id.split()
         batch_id=batch_id[3]
         with open( path+"/"+task+"/ids", "a" ) as file:
            file.write( batch_id+"\n" )

         time.sleep(1.5)


         subproc = os.fork()

         if subproc == 0:

            os.chdir( path+"/"+task )
            while True:
               nr_lines = 0
               with open( "finished_ids", "r" ) as file:
                  for line in file:
                     nr_lines += 1
               if nr_lines == total_tasks:
                  os.chdir(path+"/"+task)
                  os._exit(0)
               time.sleep(30)

         else:
            os.wait()


      os.chdir( path+"/"+task )
      content=os.listdir('.')

      conf_list=[]
      for entry in content:
         if os.path.isdir(entry) and "conformer-" in entry:
            entry=entry.split("-")
            conf_list.append( int(entry[1]) )
      conf_list.sort()

      conformers_list.clear()
      for conf in conf_list:
         os.chdir( path+"/"+task+"/conformer-"+str(conf) )
         content=os.listdir( path+"/"+task+"/conformer-"+str(conf) )
         for entry in content:
            if "ERROR" in entry or "ERROR_tmp" in entry or "tmp" in entry:
               conformers_list.append( str(conf) )

               os.mkdir("backup")
               shutil.copy( path+"/"+task+"/conformer-"+str(conf)+"/"+str(conf)+".xyz", path+"/"+task+"/conformer-"+str(conf)+"/backup/"+str(conf)+".xyz" )
               shutil.copy( path+"/"+task+"/conformer-"+str(conf)+"/run", path+"/"+task+"/conformer-"+str(conf)+"/backup/run" )
               if task == "filter_3":
                  if os.path.isfile( path+"/"+task+"/conformer-"+str(conf)+"/run_trj.xyz" ):
                     n_atoms, n_conformers=get_nr_atoms_conformers( path+"/"+task+"/conformer-"+str(conf), input_file="run_trj.xyz" )
                     extract_cart_coord( path+"/"+task+"/conformer-"+str(conf), n_atoms, input_file="run_trj.xyz", conformers_list=[ str(n_conformers) ] )
                     shutil.move( path+"/"+task+"/conformer-"+str(conf)+"/"+str(n_conformers)+".xyz", path+"/"+task+"/conformer-"+str(conf)+"/backup/run.xyz" )
                  elif os.path.isfile( path+"/"+task+"/conformer-"+str(conf)+"/run.xyz" ):
                     shutil.move( path+"/"+task+"/conformer-"+str(conf)+"/run.xyz", path+"/"+task+"/conformer-"+str(conf)+"/backup/run.xyz" )

                  with open( "run.out", "r" ) as file:
                     for line in file:
                        if "Warning: the length of the step is outside the trust region - taking restricted step instead" in line:
                           subprocess.run( [ "sed -i 's/ OPT/ COPT/' backup/run" ], shell=True, text=True )

               for filename in content:
                  file_path=os.path.join( path+"/"+task+"/conformer-"+str(conf), filename )
                  try:
                     if os.path.isfile( file_path ) or os.path.islink( file_path ):
                        os.remove( file_path )
                  except:
                     pass
               shutil.copy( path+"/"+task+"/conformer-"+str(conf)+"/backup/"+str(conf)+".xyz", path+"/"+task+"/conformer-"+str(conf)+"/"+str(conf)+".xyz" )
               shutil.copy( path+"/"+task+"/conformer-"+str(conf)+"/backup/run", path+"/"+task+"/conformer-"+str(conf)+"/run" )
               if os.path.isfile( path+"/"+task+"/conformer-"+str(conf)+"/backup/run.xyz" ):
                  shutil.copy( path+"/"+task+"/conformer-"+str(conf)+"/backup/run.xyz", path+"/"+task+"/conformer-"+str(conf)+"/run.xyz" )
               shutil.rmtree( path+"/"+task+"/conformer-"+str(conf)+"/backup" )
               if os.path.isdir( path+"/"+task+"/conformer-"+str(conf)+"/XTB" ):
                  shutil.rmtree( path+"/"+task+"/conformer-"+str(conf)+"/XTB" )
               break

      if len(conformers_list) > 0:
         restart=True

      os.chdir( path+"/"+task )

      if restart:
         with open( path+"/log", "a" ) as file:
            file.write(message+" : restart triggered\n")

      if not restart:
         if os.path.isfile( path+"/"+task+"/prev_ids" ):
            with open( path+"/"+task+"/ids", "r" ) as file:
               data = file.read()
            with open( path+"/"+task+"/prev_ids", "r" ) as file:
               data2 = file.read()
            data = data + data2
            with open( path+"/"+task+"/ids", "w" ) as file:
               file.write( data )
            subprocess.run( [ "rm -f prev_ids" ], shell=True, text=True )
         return

#-------------------------------------------------------------------------------




def timings( path, task, filter_results, task_nr=int(1) ):
#-------------------------------------------------------------------------------

   if task == "filter_1":
      ident="f1"
   elif task == "filter_2":
      ident="f2"
   elif task == "filter_3":
      ident="f3"
   elif task == "nmr_shifts":
      ident="nmr_shifts"
   elif task == "nmr_couplings":
      ident="nmr_couplings"


   os.chdir( path+"/"+task )
   batch_ids = []
   with open( path+"/"+task+"/ids", "r" ) as file:
      for line in file:
         line=line.strip()
         batch_ids.append( line )
   content=os.listdir()
   file_list = [ s for s in content for f in batch_ids if str(f) in s ]

   dft_wall_timing = [ 0, 0, 0 ]
   dft_cpu_timing  = [ 0, 0, 0 ]
   xtb_wall_timing = [ 0, 0, 0 ]
   xtb_cpu_timing  = [ 0, 0, 0 ]

   for timings_file in file_list:
      out=subprocess.run(["grep -c real "+str(timings_file)], shell=True, text=True, capture_output=True)
      out=int(out.stdout.strip())
      if ( task == "filter_1" ) or \
         ( task == "filter_2" ) or \
         ( task == "filter_3" ) :
         nr_dft = int( out / 2 )
      else:
         nr_dft = int( out )

      with open( timings_file, "r" ) as file:

         count=0
         total_wall = [ 0, 0, 0 ]
         total_cpu  = [ 0, 0, 0 ]

         for line in file:
            wall_min=0
            wall_sec=0
            wall_millisec=0
            cpu_min=0
            cpu_sec=0
            cpu_millisec=0
            if "real" in line:
               count+=1
               line=line.strip()
               line=line.split()
               wall=line[1]
               wall=wall.split("m")
               wall_min=wall[0]
               wall=wall[1].split(".")
               wall_sec=wall[0]
               wall_millisec=wall[1]
               wall_millisec=wall_millisec[:-1]
               total_wall[0] = total_wall[0] + int(wall_min)
               total_wall[1] = total_wall[1] + int(wall_sec)
               total_wall[2] = total_wall[2] + int(wall_millisec)
            if "user" in line or "sys" in line:
               line=line.strip()
               line=line.split()
               cpu=line[1]
               cpu=cpu.split("m")
               cpu_min=cpu[0]
               cpu=cpu[1].split(".")
               cpu_sec=cpu[0]
               cpu_millisec=cpu[1]
               cpu_millisec=cpu_millisec[:-1]
               total_cpu[0] = total_cpu[0] + int(cpu_min)
               total_cpu[1] = total_cpu[1] + int(cpu_sec)
               total_cpu[2] = total_cpu[2] + int(cpu_millisec)
            if count == nr_dft:
               for i in range( len(total_wall) ):
                  dft_wall_timing[i] = dft_wall_timing[i] + total_wall[i]
                  dft_cpu_timing[i] = dft_cpu_timing[i] + total_cpu[i]


               total_wall = [ 0, 0, 0 ]
               total_cpu  = [ 0, 0, 0 ]
            if count == ( nr_dft * 2 ):
               for i in range( len(total_wall) ):
                  xtb_wall_timing[i] = xtb_wall_timing[i] + total_wall[i]
                  xtb_cpu_timing[i] = xtb_cpu_timing[i] + total_cpu[i]


   with open( path+"/"+ident+"_timings", "w" ) as file:
      dft_wall=datetime.timedelta( hours=0, minutes=dft_wall_timing[0], seconds=dft_wall_timing[1], milliseconds=dft_wall_timing[2] )
      dft_cpu=datetime.timedelta(  hours=0, minutes=dft_cpu_timing[0],  seconds=dft_cpu_timing[1],  milliseconds=dft_cpu_timing[2]  )
      xtb_wall=datetime.timedelta( hours=0, minutes=xtb_wall_timing[0], seconds=xtb_wall_timing[1], milliseconds=xtb_wall_timing[2] )
      xtb_cpu=datetime.timedelta(  hours=0, minutes=xtb_cpu_timing[0],  seconds=xtb_cpu_timing[1],  milliseconds=xtb_cpu_timing[2]  )

      file.write(" DFT wall time  : {0!s:>16} ;   total seconds : {1!s:>16}\n".format( dft_wall, dft_wall.total_seconds() ) )
      file.write(" DFT cpu time   : {0!s:>16} ;   total seconds : {1!s:>16}\n".format( dft_cpu,  dft_cpu.total_seconds()  ) )
      if ( task == "filter_1" ) or ( task == "filter_2" ) or ( task == "filter_3" ):
         file.write(" XTB wall time  : {0!s:>16} ;   total seconds : {1!s:>16}\n".format( xtb_wall, xtb_wall.total_seconds() ) )
         file.write(" XTB cpu time   : {0!s:>16} ;   total seconds : {1!s:>16}\n".format( xtb_cpu,  xtb_cpu.total_seconds()  ) )

   total_wall_time = float(dft_wall.total_seconds()) + float(xtb_wall.total_seconds())
   total_cpu_time  = float(dft_cpu.total_seconds())  + float(xtb_cpu.total_seconds())

   # The task_nr is essentially only necessary (and actively used) for filter 3 (there as variable counter); in all other cases, it is simply a dummy variable
   if task_nr == 1:
      if ( task == "filter_1" ) or ( task == "filter_2" ) or ( task == "filter_3" ):
         filter_results["Timings"][task] = { "DFT_wall_time"   : dft_wall.total_seconds(),
                                             "DFT_cpu_time"    : dft_cpu.total_seconds(),
                                             "XTB_wall_time"   : xtb_wall.total_seconds(),
                                             "XTB_cpu_time"    : xtb_cpu.total_seconds(),
                                             "Total_wall_time" : total_wall_time,
                                             "Total_cpu_time"  : total_cpu_time
                                           }
      else:
         filter_results["Timings"][task] = { "Total_wall_time" : total_wall_time,
                                             "Total_cpu_time"  : total_cpu_time
                                           }
   else:
      filter_results["Timings"][task]["Total_wall_time"] = float(filter_results["Timings"][task]["Total_wall_time"]) + float (total_wall_time)
      filter_results["Timings"][task]["Total_cpu_time"]  = float(filter_results["Timings"][task]["Total_cpu_time"])  + float (total_cpu_time)
      if ( task == "filter_1" ) or ( task == "filter_2" ) or ( task == "filter_3" ):
         filter_results["Timings"][task]["DFT_wall_time"]   = float(filter_results["Timings"][task]["DFT_wall_time"])   + float (dft_wall.total_seconds())
         filter_results["Timings"][task]["DFT_cpu_time"]    = float(filter_results["Timings"][task]["DFT_cpu_time"])    + float (dft_cpu.total_seconds())
         filter_results["Timings"][task]["XTB_wall_time"]   = float(filter_results["Timings"][task]["XTB_wall_time"])   + float (xtb_wall.total_seconds())
         filter_results["Timings"][task]["XTB_cpu_time"]    = float(filter_results["Timings"][task]["XTB_cpu_time"])    + float (xtb_cpu.total_seconds())


   return
#-------------------------------------------------------------------------------




def SPEARMAN( vector_1, vector_2 ):
#-------------------------------------------------------------------------------

   if len(vector_1) != len(vector_2):
      print( " Error: intensity vector lengths do not match\n" )
      quit()

   ranking = []
   for index in range( len(vector_1) ):
      ranking.append( [ float(vector_1[index][1]), float(vector_2[index][1]), 0, 0 ] )

   ranking.sort( key = lambda x: x[0] )
   rank=1
   for index in range( len(ranking) ):
      ranking[index][2] = rank
      rank+=1

   ranking.sort( key = lambda x: x[1] )
   rank=1
   for index in range( len(ranking) ):
      ranking[index][3] = rank
      rank+=1

   sum_di_2 = 0.0
   for index in range( len(ranking) ):
      sum_di_2 = sum_di_2 + ( ( ranking[index][3] - ranking[index][2] )**2 )
   n=len(ranking)

   return ( 1.0000 - ( (6.0*sum_di_2)/(n*(n**2-1)) ) )
#-------------------------------------------------------------------------------



def bm_eval( path, wfile=None ):
#-------------------------------------------------------------------------------
   os.chdir(path)
   with open( "filter_results.json", "r" ) as file:
      filter_results=json.load( file )

   R = float( 8.31446261815324 / 1000 )
   T = 298.15

   en_bm_list = []
   # 0: conformer; 1: total energy (E + gmrrho); 2: rel. G; 3: BM factor; 4: BM weight
   E_min = 0.0

   for conf in filter_results.keys():
      if ( conf == "Molecular properties" ) or \
         ( conf == "Init. ensemble size"  ) or \
         ( conf == "Solvent"              ) or \
         ( conf == "Level"                ) or \
         ( conf == "Reference"            ) or \
         ( conf == "Timings"              ) :
         continue
      if filter_results[conf]["filter_3"]["converged"] == "true" :
         total_G = float( filter_results[conf]["filter_3"]["DFT_energy"] ) + float( filter_results[conf]["filter_3"]["XTB_gmrrho"] )
         en_bm_list.append( [ conf, total_G ] )

   en_bm_list = sorted( en_bm_list, key = lambda x: x[1] )

   sum_bm_factors = 0.0
   for conf in range( len(en_bm_list) ):
      if conf == 0:
         en_bm_list[conf].append( 0.0 )
      else:
         en_bm_list[conf].append( ( en_bm_list[conf][1] - en_bm_list[0][1] ) * ha_to_kcal * kcal_to_kj )
      en_bm_list[conf].append( math.exp( -(en_bm_list[conf][2]/(R*T)) ) )
      sum_bm_factors += en_bm_list[conf][3]

   for conf in range( len(en_bm_list) ):
      en_bm_list[conf].append( ( en_bm_list[conf][3]/sum_bm_factors )*100 )

   n_atoms = filter_results["Molecular properties"]["Atoms"]
   n_conformers = filter_results["Init. ensemble size"]
   nr_conf_G = 0
   fract_BM_sum_G = 0.0
   for conf in range( len(en_bm_list) ):
      if en_bm_list[conf][2] > thresh_G:
         break
      nr_conf_G += 1
      fract_BM_sum_G += en_bm_list[conf][4]

   nr_conf_BM = 0
   fract_BM_sum_BM = 0.0
   max_G = 20.00
   for conf in range( len(en_bm_list) ):
      if fract_BM_sum_BM > thresh_BM :
         nr_conf_BM -= 1
         fract_BM_sum_BM -= en_bm_list[conf-1][4]
         max_G = en_bm_list[conf-2][2]
         break
      nr_conf_BM += 1
      fract_BM_sum_BM += en_bm_list[conf][4]

   if nr_conf_G < nr_conf_BM:
      nr_conf = nr_conf_G
   else:
      nr_conf = nr_conf_BM

   if wfile is not None:
      with open( wfile, "w" ) as file:
         file.write( "Initial conformational ensemble size     : {}\n".format( n_conformers ) )
         file.write( "Conformers within threshold of 20 kJ/mol : {}\n".format( len(en_bm_list) ) )
         file.write( "Maximum relative free energy difference  : {:<5.2f} kJ/mol\n".format( en_bm_list[-1][2] ) )
         file.write( "\nThresholds for subsequent processing:\n" )
         file.write( "dG_rel    : {0:>6.2f} kJ/mol  --> {1:>5d} conformers ( {2:>6.2f} % wgt. of BM sum )\n".format( thresh_G, nr_conf_G, fract_BM_sum_G ) )
         file.write( "frac. BM  : {0:>6.2f} % wgt.  --> {1:>5d} conformers ( {2:>6.2f} kJ/mol dG_rel    )\n".format( thresh_BM, nr_conf_BM, max_G ) )

      print( "\nInitial conformational ensemble size     : {}".format( n_conformers ) )
      print( "Conformers within threshold of 20 kJ/mol : {}".format( len(en_bm_list) ) )
      print( "Maximum relative free energy difference  : {:<5.2f} kJ/mol".format( en_bm_list[-1][2] ) )
      print( "\nThresholds for subsequent processing:" )
      print( "  dG_rel    : {0:>6.2f} kJ/mol  --> {1:>5d} conformers ( {2:>6.2f} % wgt. of BM sum )".format( thresh_G, nr_conf_G, fract_BM_sum_G ) )
      print( "  frac. BM  : {0:>6.2f} % wgt.  --> {1:>5d} conformers ( {2:>6.2f} kJ/mol dG_rel    )".format( thresh_BM, nr_conf_BM, max_G ) )

   return en_bm_list, nr_conf_G, nr_conf_BM
#-------------------------------------------------------------------------------




def packing( path, stp ):
#-------------------------------------------------------------------------------

   if os.path.isdir( path+"/"+stp ):
      if os.path.isfile( path+"/"+stp+".tar.gz" ): # Remove potentially existing old tar.gz file
         os.remove( path+"/"+stp+".tar.gz" )
      subprocess.run(
                      [ "if [ -d "+path+"/"+stp+" ]; then if ! [ -e "+path+"/"+stp+".tar.gz ]; then tar cvfz "+stp+".tar.gz "+stp+" &> /dev/null; fi; fi;" ],
                      shell=True, text=True
                    )
      shutil.rmtree( path+"/"+stp )
   else:
      print("\n Error: "+stp+" data directory not found\n")
      quit()

   return
#-------------------------------------------------------------------------------




def unpacking( path, stp ):
#-------------------------------------------------------------------------------

   if os.path.isfile( path+"/"+stp+".tar.gz" ):
      if os.path.isdir( path+"/"+stp ): # Remove potentially existing old directory
         shutil.rmtree( path+"/"+stp )
      subprocess.run(
                      [ "if ! [ -d "+path+"/"+stp+" ]; then if [ -e "+path+"/"+stp+".tar.gz ]; then tar xvfz "+path+"/"+stp+".tar.gz &> /dev/null; fi; fi;" ],
                      shell=True, text=True
                    )
      os.remove( path+"/"+stp+".tar.gz" )
   else:
      print("\n Error: "+stp+".tar.gz data file not found\n")
      quit()

   return
#-------------------------------------------------------------------------------




def filter_1( path ):
#-------------------------------------------------------------------------------

   os.chdir(path)
   os.mkdir(path+"/filter_1")
   os.chdir(path+"/filter_1")

   with open("run", "w") as file:
      file.write(filter_1_input)

   n_atoms, n_conformers=get_nr_atoms_conformers( path )
   extract_cart_coord( path, n_atoms )

   conformers=[]
   for conf in range(1, n_conformers+1):
      conformers.append( str(conf) )
      os.mkdir( path+"/filter_1/conformer-"+str(conf) )
      shutil.copy( path+"/filter_1/run", path+"/filter_1/conformer-"+str(conf)+"/run" )
      shutil.copy( path+"/filter_1/"+str(conf)+".xyz", path+"/filter_1/conformer-"+str(conf)+"/"+str(conf)+".xyz" )


   nnodes, ntasks, mem, time, parti, qos, prog, xtb_home, email, opt, xtb, xtb_hess = batch_settings( "filter_1" )
   source_file="batch_array"
   with open( "slurm_template", "w" ) as file:
      file.write( batch_file.format( NODES=nnodes, TASKS=ntasks, MEM=mem, TIME=time, PARTI=parti, QOS=qos, MAIL=email,                     \
                                     PROG=prog, XTB=xtb_home, MULTI=multi, CHARGE=charge, SOLV_XTB=solvent_XTB, XTB_SMODEL=XTB_solv_model, \
                                     OPT_BOOL=opt, XTB_BOOL=xtb, SOURCE=source_file, XTB_HESS=xtb_hess                                     ) )


   cpy_conformers = copy.deepcopy( conformers )
   comp_driver( path, "filter_1", cpy_conformers )


   # Evaluation


   filter_results = {
                    "Molecular properties" : {
                                             "Atoms"        : n_atoms,
                                             "Charge"       : charge,
                                             "Multiplicity" : multi,
                                             "CREST minimum conformer" : {}
                                             },
                    "Init. ensemble size"  : n_conformers,
                    "Solvent"              : solvent_ORCA,
                    "Level"                : "",
                    "Reference"            : ""
                    }

   filter_results["Reference"] = { solvent_ORCA : "" }
   filter_results["Reference"][solvent_ORCA] =  References[func_shifts+"/pcSseg-3"][solvent_ORCA]

   if not os.path.isfile( "1.xyz" ):
      extract_cart_coord( source, n_atoms, conformers_list=[ str(1) ] )
   with open( "1.xyz", "r" ) as file:
      lnr = 0
      anr = 1
      for line in file:
         lnr += 1
         if lnr <= 2:
            continue
         line.strip()
         line = line.split()
         filter_results["Molecular properties"]["CREST minimum conformer"][anr] = line[0]+" "+line[1]+" "+line[2]+" "+line[3]
         anr += 1


   for conf in range(1, n_conformers+1):
      filter_results[str(conf)] = { "filter_1" : { "DFT_energy" : "", "XTB_gsolv" : "", "E_rel" : "", "eligible" : "false" } }

   filter_results["Timings"] = { "filter_1" : { "DFT_wall_time"   : "",
                                                "DFT_cpu_time"    : "",
                                                "XTB_wall_time"   : "",
                                                "XTB_cpu_time"    : "",
                                                "Total_wall_time" : "",
                                                "Total_cpu_time"  : ""
                                              }
                               }

   # Evaluate timings
   timings( path, "filter_1", filter_results )

   gsolv=0.0
   for conf in filter_results.keys():

      if ( conf == "Molecular properties" ) or \
         ( conf == "Init. ensemble size"  ) or \
         ( conf == "Solvent"              ) or \
         ( conf == "Level"                ) or \
         ( conf == "Reference"            ) or \
         ( conf == "Timings"              ) :
         continue

      os.chdir(path+"/filter_1/conformer-"+str(conf))
      with open("run.out", "r") as file:
         for line in file:
            if "FINAL SINGLE POINT ENERGY" in line:
               line=line.strip()
               line=line.split()
               dft=float(line[4])
      os.chdir(path+"/filter_1/conformer-"+str(conf)+"/XTB")
      with open("XTB.out", "r") as file:
         for line in file:
            if "Gsolv" in line:
               line=line.strip()
               line=line.split()
               gsolv=float(line[3])
      filter_results[str(conf)]["filter_1"]["DFT_energy"] = dft
      filter_results[str(conf)]["filter_1"]["XTB_gsolv"]  = gsolv
      os.chdir(path)

   min_conf=0
   min_e=float(0)
   for conf in filter_results.keys():

      if ( conf == "Molecular properties" ) or \
         ( conf == "Init. ensemble size"  ) or \
         ( conf == "Solvent"              ) or \
         ( conf == "Level"                ) or \
         ( conf == "Reference"            ) or \
         ( conf == "Timings"              ) :
         continue

      if ( filter_results[conf]["filter_1"]["DFT_energy"] + filter_results[conf]["filter_1"]["XTB_gsolv"] ) < min_e:
         min_e = ( filter_results[conf]["filter_1"]["DFT_energy"] + filter_results[conf]["filter_1"]["XTB_gsolv"] )
         min_conf = conf

   for conf in filter_results.keys():
      if ( conf == "Molecular properties" ) or \
         ( conf == "Init. ensemble size"  ) or \
         ( conf == "Solvent"              ) or \
         ( conf == "Level"                ) or \
         ( conf == "Reference"            ) or \
         ( conf == "Timings"              ) :
         continue

      E_rel = ( ( filter_results[conf]["filter_1"]["DFT_energy"] + filter_results[conf]["filter_1"]["XTB_gsolv"] - min_e ) * ha_to_kcal * kcal_to_kj )
      filter_results[conf]["filter_1"]["E_rel"] = "{E:9.4f}".format( E=E_rel )
      if E_rel <= filter_1_thr:
         filter_results[conf]["filter_1"]["eligible"] = "true"

   os.chdir(path)

   with open( "filter_results.json", "w" ) as file:
      json.dump(filter_results, file, indent=4)

   with open( "log", "a" ) as file:
      file.write("filter 1 done\n")


# Write results to file
   n_atoms, n_conformers = get_nr_atoms_conformers( source )
   counter_f1=0
   for conf in filter_results.keys():
      if ( conf == "Molecular properties" ) or \
         ( conf == "Init. ensemble size"  ) or \
         ( conf == "Solvent"              ) or \
         ( conf == "Level"                ) or \
         ( conf == "Reference"            ) or \
         ( conf == "Timings"              ) :
         continue

      if filter_results[conf]["filter_1"]["eligible"] == "true":
         counter_f1+=1

   with open( "progress.out", "w" ) as file:
      file.write("Initial    conformrs: {0!s:>5s}\n".format(n_conformers))
      file.write("Filter 1   reduction: {0!s:>5s}  --> {1!s:>5s}  [thr <= {thr1!s}]\n".format(n_conformers, counter_f1, thr1=filter_1_thr))

   return
#-------------------------------------------------------------------------------




def filter_2( path ):
#-------------------------------------------------------------------------------


   os.chdir(path)

   with open( path+"/filter_results.json", "r" ) as file:
      filter_results=json.load( file )
   conformers = []
   for conf in filter_results.keys():
      if ( conf == "Molecular properties" ) or \
         ( conf == "Init. ensemble size"  ) or \
         ( conf == "Solvent"              ) or \
         ( conf == "Level"                ) or \
         ( conf == "Reference"            ) or \
         ( conf == "Timings"              ) :
         continue
      if filter_results[conf]["filter_1"]["eligible"] == "true":
         conformers.append( str(conf) )

   os.mkdir(path+"/filter_2")
   os.chdir(path+"/filter_2")

   with open("run", "w") as file:
      file.write(filter_2_input)

   n_atoms, n_conformers=get_nr_atoms_conformers( path )
   extract_cart_coord( path, n_atoms, conformers_list=conformers )

   for conf in conformers:
      os.mkdir( path+"/filter_2/conformer-"+str(conf) )
      shutil.copy( path+"/filter_2/run", path+"/filter_2/conformer-"+str(conf)+"/run" )
      shutil.copy( path+"/filter_2/"+str(conf)+".xyz", path+"/filter_2/conformer-"+str(conf)+"/"+str(conf)+".xyz" )


   nnodes, ntasks, mem, time, parti, qos, prog, xtb_home, email, opt, xtb, xtb_hess = batch_settings( "filter_2" )
   source_file="batch_array"
   with open( "slurm_template", "w" ) as file:
      file.write( batch_file.format( NODES=nnodes, TASKS=ntasks, MEM=mem, TIME=time, PARTI=parti, QOS=qos, MAIL=email,                     \
                                     PROG=prog, XTB=xtb_home, MULTI=multi, CHARGE=charge, SOLV_XTB=solvent_XTB, XTB_SMODEL=XTB_solv_model, \
                                     OPT_BOOL=opt, XTB_BOOL=xtb, SOURCE=source_file, XTB_HESS=xtb_hess                                     ) )

   cpy_conformers = copy.deepcopy( conformers )
   comp_driver( path, "filter_2", cpy_conformers )



   # Evaluation
   for conf in filter_results.keys():
      if ( conf == "Molecular properties" ) or \
         ( conf == "Init. ensemble size"  ) or \
         ( conf == "Solvent"              ) or \
         ( conf == "Level"                ) or \
         ( conf == "Reference"            ) or \
         ( conf == "Timings"              ) :
         continue
      filter_results[str(conf)]["filter_2"] = { "DFT_energy" : "",
                                                "XTB_gmrrho" : "",
                                                "E_rel"      : "",
                                                "eligible"   : "false"
                                              }


   # Evaluate timings
   timings( path, "filter_2", filter_results )


   gmrrho=0.0
   for conf in conformers:
      os.chdir(path+"/filter_2/conformer-"+str(conf))
      with open("run.out", "r") as file:
         for line in file:
            if "FINAL SINGLE POINT ENERGY" in line:
               line=line.strip()
               line=line.split()
               dft=float(line[4])
      os.chdir(path+"/filter_2/conformer-"+str(conf)+"/XTB")
      with open("XTB.out", "r") as file:
         for line in file:
            if "G(RRHO) contrib." in line:
               line=line.strip()
               line=line.split()
               gmrrho=float(line[3])
      filter_results[str(conf)]["filter_2"]["DFT_energy"] = dft
      filter_results[str(conf)]["filter_2"]["XTB_gmrrho"] = gmrrho
      os.chdir(path)

   min_conf=0
   min_e=float(0)
   mrrho_array = []
   for conf in filter_results.keys():
      if ( conf == "Molecular properties" )                       or \
         ( conf == "Init. ensemble size"  )                       or \
         ( conf == "Solvent"              )                       or \
         ( conf == "Level"                )                       or \
         ( conf == "Reference"            )                       or \
         ( conf == "Timings"              )                       or \
         ( filter_results[conf]["filter_2"]["DFT_energy"] == "" ) :
         continue

      if ( filter_results[conf]["filter_2"]["DFT_energy"] + filter_results[conf]["filter_2"]["XTB_gmrrho"] ) < min_e:
         min_e = ( filter_results[conf]["filter_2"]["DFT_energy"] + filter_results[conf]["filter_2"]["XTB_gmrrho"] )
         min_conf = conf
      mrrho_array.append( filter_results[conf]["filter_2"]["XTB_gmrrho"] )

   std_mrrho=np.std( mrrho_array )
   for conf in filter_results.keys():

      if ( conf == "Molecular properties" )                       or \
         ( conf == "Init. ensemble size"  )                       or \
         ( conf == "Solvent"              )                       or \
         ( conf == "Level"                )                       or \
         ( conf == "Reference"            )                       or \
         ( conf == "Timings"              )                       or \
         ( filter_results[conf]["filter_2"]["DFT_energy"] == "" ) :
         continue

      E_rel = ( ( filter_results[conf]["filter_2"]["DFT_energy"] + filter_results[conf]["filter_2"]["XTB_gmrrho"] - min_e ) * ha_to_kcal * kcal_to_kj )
      filter_results[conf]["filter_2"]["E_rel"] = "{E:9.4f}".format( E=E_rel )
      if E_rel <= ( filter_2_thr + ( float(std_mrrho) * ha_to_kcal * kcal_to_kj ) ):
         filter_results[conf]["filter_2"]["eligible"] = "true"

   os.chdir(path)

   with open( "filter_results.json", "w" ) as file:
      json.dump(filter_results, file, indent=4)

   with open( "log", "a" ) as file:
      file.write("filter 2 done\n")

# Write results to file
   counter_f1=0
   counter_f2=0
   for conf in filter_results.keys():
      if ( conf == "Molecular properties" ) or \
         ( conf == "Init. ensemble size"  ) or \
         ( conf == "Solvent"              ) or \
         ( conf == "Level"                ) or \
         ( conf == "Reference"            ) or \
         ( conf == "Timings"              ) :
         continue
      if filter_results[conf]["filter_1"]["eligible"] == "true":
         counter_f1+=1
      if filter_results[conf]["filter_2"]["eligible"] == "true":
         counter_f2+=1

   with open( "progress.out", "a" ) as file:
      file.write("Filter 2   reduction: {0!s:>5s}  --> {1!s:>5s}  [thr <= {thr2!s}]\n".format(counter_f1, counter_f2, thr2=filter_2_thr))

   return
#-------------------------------------------------------------------------------




def filter_3( path, counter ):
#-------------------------------------------------------------------------------

   os.chdir(path)
   with open( "filter_results.json", "r" ) as file:
      filter_results=json.load( file )

   conformers = []
   if counter == 1:

      for conf in filter_results.keys():
         if ( conf == "Molecular properties" ) or \
            ( conf == "Init. ensemble size"  ) or \
            ( conf == "Solvent"              ) or \
            ( conf == "Level"                ) or \
            ( conf == "Reference"            ) or \
            ( conf == "Timings"              ) :
            continue

         filter_results[conf]["filter_3"] = { "DFT_energy" : "",
                                              "XTB_gmrrho" : "",
                                              "E_rel"      : "",
                                              "converged"  : "false",
                                              "eligible"   : "true"
                                            }

         if filter_results[conf]["filter_2"]["eligible"] == "true":
            conformers.append( conf )
         elif filter_results[conf]["filter_2"]["eligible"] == "false":
            filter_results[conf]["filter_3"]["eligible"] = "false"

      os.mkdir(path+"/filter_3")
      os.chdir(path+"/filter_3")

      with open("run", "w") as file:
         file.write(filter_3_input)

      n_atoms, n_conformers=get_nr_atoms_conformers( path )
      extract_cart_coord( path, n_atoms, conformers_list=conformers )

      for conf in conformers:
         os.mkdir( path+"/filter_3/conformer-"+str(conf) )
         shutil.copy( path+"/filter_3/run", path+"/filter_3/conformer-"+str(conf)+"/run" )
         shutil.copy( path+"/filter_3/"+str(conf)+".xyz", path+"/filter_3/conformer-"+str(conf)+"/"+str(conf)+".xyz" )
         shutil.copy( path+"/filter_3/"+str(conf)+".xyz", path+"/filter_3/conformer-"+str(conf)+"/run.xyz" )

      nnodes, ntasks, mem, time, parti, qos, prog, xtb_home, email, opt, xtb, xtb_hess = batch_settings( "filter_3" )
      source_file="batch_array"
      with open( "slurm_template", "w" ) as file:
         file.write( batch_file.format( NODES=nnodes, TASKS=ntasks, MEM=mem, TIME=time, PARTI=parti, QOS=qos, MAIL=email,                     \
                                        PROG=prog, XTB=xtb_home, MULTI=multi, CHARGE=charge, SOLV_XTB=solvent_XTB, XTB_SMODEL=XTB_solv_model, \
                                        OPT_BOOL=opt, XTB_BOOL=xtb, SOURCE=source_file, XTB_HESS=xtb_hess                                     ) )


   else:

      for conf in filter_results.keys():
         if ( conf == "Molecular properties" ) or \
            ( conf == "Init. ensemble size"  ) or \
            ( conf == "Solvent"              ) or \
            ( conf == "Level"                ) or \
            ( conf == "Reference"            ) or \
            ( conf == "Timings"              ) :
            continue

         if ( filter_results[conf]["filter_3"]["converged"] == "false" ) and \
            ( filter_results[conf]["filter_3"]["eligible"]  == "true"  ) :
            conformers.append( conf )

   os.chdir(path+"/filter_3")


   # Start calculations
   cpy_conformers = copy.deepcopy( conformers )
   comp_driver( path, "filter_3", cpy_conformers )

   # Evaluate timing
   timings( path, "filter_3", filter_results, counter )

   # Evaluate results
   os.chdir(path)
   pes={}
   for conf in conformers:
      os.chdir(path+"/filter_3/conformer-"+str(conf))
      converged = False
      gmrrho=0.0
      with open( "run.out", "r" ) as file:
         for line in file:
            if "THE OPTIMIZATION HAS CONVERGED" in line:
               converged = True
               filter_results[conf]["filter_3"]["converged"] = "true"
            if converged:
               if "FINAL SINGLE POINT ENERGY" in line:
                  line=line.strip()
                  line=line.split()
                  filter_results[conf]["filter_3"]["DFT_energy"] = float( line[4] )
               os.chdir(path+"/filter_3/conformer-"+str(conf)+"/XTB")
               with open("XTB.out", "r") as file2:
                  for line2 in file2:
                     if "G(RRHO) contrib." in line2:
                        line2=line2.strip()
                        line2=line2.split()
                        gmrrho=line2[3]
                        break
               filter_results[conf]["filter_3"]["XTB_gmrrho"] = float(gmrrho)
               os.chdir(path+"/filter_3/conformer-"+str(conf))


      if not converged:
         pes[conf] = []
         with open( "run_trj.xyz", "r" ) as file:
            for line in file:
               if "Coordinates" in line:
                  line=line.strip()
                  line=line.split()
                  energy=line[5]
                  pes[conf].append( energy )
         if len(pes[conf]) != 8:
            del pes[conf]
            continue
      os.chdir( path+"/filter_3" )


# Evaluate converged optimizations


   os.chdir( path )
   min_conf=0
   min_e=float(0)
   for conf in filter_results.keys():

      if ( conf == "Molecular properties" )                           or \
         ( conf == "Init. ensemble size"  )                           or \
         ( conf == "Solvent"              )                           or \
         ( conf == "Level"                )                           or \
         ( conf == "Reference"            )                           or \
         ( conf == "Timings"              )                           or \
         ( filter_results[conf]["filter_3"]["converged"] == "false" ) or \
         ( filter_results[conf]["filter_3"]["eligible"]  == "false" ) :
         continue

      if ( filter_results[conf]["filter_3"]["DFT_energy"] + filter_results[conf]["filter_3"]["XTB_gmrrho"] ) < min_e:
         min_e = ( filter_results[conf]["filter_3"]["DFT_energy"] + filter_results[conf]["filter_3"]["XTB_gmrrho"] )
         min_conf = conf


   for conf in filter_results.keys():

      if ( conf == "Molecular properties" )                           or \
         ( conf == "Init. ensemble size"  )                           or \
         ( conf == "Solvent"              )                           or \
         ( conf == "Level"                )                           or \
         ( conf == "Reference"            )                           or \
         ( conf == "Timings"              )                           or \
         ( filter_results[conf]["filter_3"]["converged"] == "false" ) or \
         ( filter_results[conf]["filter_3"]["eligible"] == "false"  ) :
         continue

      E_rel = ( ( filter_results[conf]["filter_3"]["DFT_energy"] + filter_results[conf]["filter_3"]["XTB_gmrrho"] - min_e ) * ha_to_kcal * kcal_to_kj )
      filter_results[conf]["filter_3"]["E_rel"] = "{E:9.4f}".format( E=E_rel )
      if E_rel > filter_3_thr:
         filter_results[conf]["filter_3"]["eligible"] = "false"




# TEST FOR PARALLELITY

   for conformer1 in filter_results.keys():

      eval_list={}
      if counter == 1:
         thr = 20.0
      elif counter == 2:
         thr = 15.0
      else:
         thr = 10.0


      if ( conf == "Molecular properties" )                                 or \
         ( conf == "Init. ensemble size"  )                                 or \
         ( conf == "Solvent"              )                                 or \
         ( conf == "Level"                )                                 or \
         ( conf == "Reference"            )                                 or \
         ( conf == "Timings"              )                                 or \
         ( filter_results[conformer1]["filter_3"]["converged"] == "true"  ) or \
         ( filter_results[conformer1]["filter_3"]["eligible"]  == "false" ) :
         continue


      opt_cycles_c1 = []
      try:
         for i in range( 1, 9 ):
            opt_cycles_c1.append( [ int(i), float(pes[conformer1][i-1]) ] )
      except KeyError:
         continue


      for conformer2 in filter_results.keys():

         if ( conf == "Molecular properties" )                                 or \
            ( conf == "Init. ensemble size"  )                                 or \
            ( conf == "Solvent"              )                                 or \
            ( conf == "Level"                )                                 or \
            ( conf == "Reference"            )                                 or \
            ( conf == "Timings"              )                                 or \
            ( filter_results[conformer2]["filter_3"]["converged"] == "true"  ) or \
            ( filter_results[conformer2]["filter_3"]["eligible"]  == "false" ) or \
            ( conformer1 == conformer2 )                                       :
            continue

         opt_cycles_c2 = []
         try:
            for i in range( 1, 9 ):
               opt_cycles_c2.append( [ int(i), float(pes[conformer2][i-1]) ] )
         except KeyError:
            continue

         corr = SPEARMAN( opt_cycles_c1, opt_cycles_c2 )
         if corr >= 0.92:
            eval_list[conformer2] = float(pes[conformer2][-1])
            os.chdir(path+"/filter_3/conformer-"+str(conformer2)+"/XTB")
            with open("XTB.out", "r") as file:
               for line in file:
                  if "G(RRHO) contrib." in line:
                     line=line.strip()
                     line=line.split()
                     gmrrho=line[3]
                     break
            eval_list[conformer2] = ( float(eval_list[conformer2]) + float(gmrrho) )
            os.chdir(path)


      if len(eval_list) >= 1:
         eval_list[conformer1] = float(pes[conformer1][-1])
         os.chdir(path+"/filter_3/conformer-"+str(conformer1)+"/XTB")
         with open("XTB.out", "r") as file:
            for line in file:
               if "G(RRHO) contrib." in line:
                  line=line.strip()
                  line=line.split()
                  gmrrho=line[3]
                  break
         eval_list[conformer1] = ( float(eval_list[conformer1]) + float(gmrrho) )
         os.chdir(path)

         min_conf = 0
         min_e = 0.0
         for entry in eval_list.keys():
            if eval_list[entry] < min_e:
               min_e = eval_list[entry]
               min_conf = entry


         for entry in eval_list.keys():
            if ( ( eval_list[entry] - eval_list[min_conf] ) * ha_to_kcal * kcal_to_kj ) > thr :
               os.chdir(path+"/filter_3/conformer-"+str(entry))
               with open( "run.out", "r" ) as file:
                  for line in file:
                     if "RMS gradient" in line:
                        line=line.strip()
                        line=line.split()
                        grad_norm = 1.0
                        try:
                           grad_norm = float(line[2])
                        except ValueError:
                           pass
               os.chdir(path)
               if grad_norm < 0.01:
                  filter_results[entry]["filter_3"]["eligible"] = "false"


   os.chdir(path)
   with open( "filter_results.json", "w" ) as file:
      json.dump(filter_results, file, indent=4)
   with open( "log", "a" ) as file:
      file.write("filter 3 "+str(counter)+" done\n")


# Write results to file
   counter_f2=0
   counter_f3=0
   counter_f3_spearman=0
   counter_f3_converged=0
   counter_f3_converged_elig=0
   for conf in filter_results.keys():
      if ( conf == "Molecular properties" ) or \
         ( conf == "Init. ensemble size"  ) or \
         ( conf == "Solvent"              ) or \
         ( conf == "Level"                ) or \
         ( conf == "Reference"            ) or \
         ( conf == "Timings"              ) :
         continue
      if filter_results[conf]["filter_2"]["eligible"] == "true":
         counter_f2+=1
      if filter_results[conf]["filter_3"]["eligible"] == "true":
         counter_f3+=1
      if ( filter_results[conf]["filter_3"]["eligible"] == "false" ) and ( filter_results[conf]["filter_3"]["converged"] == "false" ):
         if filter_results[conf]["filter_2"]["eligible"] == "true":
            counter_f3_spearman+=1
      if ( filter_results[conf]["filter_3"]["converged"] == "true" ) and ( filter_results[conf]["filter_3"]["eligible"] == "false" ):
         counter_f3_converged+=1
      if ( filter_results[conf]["filter_3"]["converged"] == "true" ) and ( filter_results[conf]["filter_3"]["eligible"] == "true" ):
         counter_f3_converged_elig+=1

# read actual filter counter
   if counter == 1:
      with open( "progress.out", "a" ) as file:
         file.write("Filter 3-{loop!s} reduction: {0!s:>5s}  --> {1!s:>5s}  [thr <= {thr3!s}]  ( conv < thr : {thresh!s:>4s},  conv > thr : {threshout!s:>4s},  spear > thr : {spearman!s} )\n".format(counter_f2, counter_f3, loop=counter, thr3=thr, thresh=counter_f3_converged_elig, threshout=counter_f3_converged, spearman=counter_f3_spearman))

   else:
      with open( "progress.out", "r" ) as file:
         for line in file:
            if "Filter 3" in line:
               line=line.strip()
               line=line.split()
               actual_counter_f3=line[5]
      with open( "progress.out", "a" ) as file:
         file.write("Filter 3-{loop!s} reduction: {0!s:>5s}  --> {1!s:>5s}  [thr <= {thr3!s}]  ( conv < thr : {thresh!s:>4s},  conv > thr : {threshout!s:>4s},  spear > thr : {spearman!s} )\n".format(actual_counter_f3, counter_f3, loop=counter, thr3=thr, thresh=counter_f3_converged_elig, threshout=counter_f3_converged, spearman=counter_f3_spearman))

   return
#-------------------------------------------------------------------------------




def filter_3_driver( path ):
#-------------------------------------------------------------------------------

   if restart_f3_cycle is None:
      cycle = 1
   else:
      cycle = restart_f3_cycle

   while True:

      filter_3( source, cycle )

      os.chdir( path )
      with open( "filter_results.json", "r" ) as file:
         filter_results=json.load( file )

      finished = True
      for conf in filter_results.keys():
         if ( conf == "Molecular properties" ) or \
            ( conf == "Init. ensemble size"  ) or \
            ( conf == "Solvent"              ) or \
            ( conf == "Level"                ) or \
            ( conf == "Reference"            ) or \
            ( conf == "Timings"              ) :
            continue
         if ( filter_results[conf]["filter_3"]["converged"] == "false" ) and ( filter_results[conf]["filter_3"]["eligible"] == "true" ):
            finished = False
      if finished:
         break

      cycle += 1
      if cycle == 100:
         break


   # Write results to file 'progrss.out'
   counter_f2=0
   counter_f3=0
   counter_f3_spearman=0
   counter_f3_converged=0
   counter_f3_converged_elig=0
   for conf in filter_results.keys():
      if ( conf == "Molecular properties" ) or \
         ( conf == "Init. ensemble size"  ) or \
         ( conf == "Solvent"              ) or \
         ( conf == "Level"                ) or \
         ( conf == "Reference"            ) or \
         ( conf == "Timings"              ) :
         continue
      if filter_results[conf]["filter_2"]["eligible"] == "true":
         counter_f2+=1
      if filter_results[conf]["filter_3"]["eligible"] == "true":
         counter_f3+=1
      if ( filter_results[conf]["filter_3"]["eligible"] == "false" ) and ( filter_results[conf]["filter_3"]["converged"] == "false" ):
         if filter_results[conf]["filter_2"]["eligible"] == "true":
            counter_f3_spearman+=1
      if ( filter_results[conf]["filter_3"]["converged"] == "true" ) and ( filter_results[conf]["filter_3"]["eligible"] == "false" ):
         counter_f3_converged+=1
      if ( filter_results[conf]["filter_3"]["converged"] == "true" ) and ( filter_results[conf]["filter_3"]["eligible"] == "true" ):
         counter_f3_converged_elig+=1

   with open( "progress.out", "a" ) as file:
      file.write( "Filter 3   reduction: {0!s:>5s}  --> {1!s:>5s}  [thr <= {thr3!s}]  ( conv < thr : {thresh!s:>4s},  conv > thr : {threshout!s:>4s},  spear > thr : {spearman!s} )\n".format( \
                  counter_f2, counter_f3, thr3=filter_3_thr, thresh=counter_f3_converged_elig, threshout=counter_f3_converged, spearman=counter_f3_spearman                                    ) )


   return
#-------------------------------------------------------------------------------




def conf_gen( path ):
#-------------------------------------------------------------------------------

   os.chdir(path)

   en_bm_list, nr_conf_G, nr_conf_BM = bm_eval( source, "sorting.out" )

   print(en_bm_list)
   print("-----")
   print(nr_conf_G)
   print("-----")
   print(nr_conf_BM)

   sys.exit()

   if nr_conf_G < nr_conf_BM:
      nr_conf = nr_conf_G
   else:
      nr_conf = nr_conf_BM
   print("\nChoosing lowest number of conformers for sorting process ...\n")

   with open( "filter_results.json", "r" ) as file:
      filter_results=json.load( file )

   os.mkdir( path+"/sorting" )
   os.chdir( path+"/sorting" )

   with open ( "filter_ensemble.xyz", "w" ) as file_results:

      for conf in range( 0, nr_conf ):
         if filter_results[en_bm_list[conf][0]]["filter_3"]["converged"] == "true" :
            os.chdir( path+"/filter_3/conformer-"+str(en_bm_list[conf][0]) )
            line_nr=1
            with open( "run.xyz", "r" ) as file_coords:
               for line in file_coords:
                  line=line.strip()
                  if line_nr == 2:
                     file_results.write( str( float(filter_results[en_bm_list[conf][0]]["filter_3"]["DFT_energy"]) + \
                                              float(filter_results[en_bm_list[conf][0]]["filter_3"]["XTB_gmrrho"]) )+"\n" )
                  else:
                     file_results.write( line+"\n" )
                  line_nr += 1

      os.chdir( path+"/sorting" )

   n_atoms, n_conformers=get_nr_atoms_conformers( path )
   extract_cart_coord( source, n_atoms, conformers_list=[ str(1) ] )
   subprocess.run( [ "crest 1.xyz --cregen filter_ensemble.xyz | tee CREGEN.out" ], shell=True, capture_output=True, text=True )
      
   os.chdir( path )
   with open( "log", "a" ) as file:
      file.write( "conformer processing done\n" )


   return
#-------------------------------------------------------------------------------




def nmr_shifts( path ):
#-------------------------------------------------------------------------------

   os.chdir(path)

   en_bm_list, nr_conf_G, nr_conf_BM = bm_eval( source, "sorting_shifts.out" )

   if nr_conf_G < nr_conf_BM:
      nr_conf = nr_conf_G
   else:
      nr_conf = nr_conf_BM
   print("\nChoosing lowest number of conformers for sorting process ...\n")

   # Load json file to write timings to it later on
   with open( "filter_results.json", "r" ) as file:
      filter_results=json.load( file )

   os.mkdir( path+"/nmr_shifts" )

   conformers = []
   with open( "anmr_enso", "w" ) as file:
      file.write( "ONOFF NMR  CONF BW      Energy       Gsolv      mRRHO      gi\n" )
      for conf in range( 0, nr_conf ):
         if filter_results[en_bm_list[conf][0]]["filter_3"]["converged"] == "true" :
            os.mkdir( path+"/nmr_shifts/conformer-"+str(en_bm_list[conf][0]) )
            shutil.copy( path+"/filter_3/conformer-"+str(en_bm_list[conf][0])+"/run.xyz", path+"/nmr_shifts/conformer-"+str(en_bm_list[conf][0])+"/"+str(en_bm_list[conf][0])+".xyz" )
            file.write( "1  {0:>5}{1:>5}    {2:6.4f} {3:<13.7f} 0.0000000  0.0000000 1.000\n".format( str(en_bm_list[conf][0]), str(en_bm_list[conf][0]), float(en_bm_list[conf][4])/100, float(en_bm_list[conf][1]) ) )
         conformers.append( en_bm_list[conf][0] )


   filter_results["Level"] = func_shifts+"/pcSseg3"
   filter_results["Reference"] = { solvent_ORCA : "" }
   filter_results["Reference"][solvent_ORCA] =  References[func_shifts+"/pcSseg-3"][solvent_ORCA] #: References[filter_results]["Level"][solvent_ORCA] }


   for conf in conformers:
      os.chdir( path+"/nmr_shifts/conformer-"+str(conf) )
      with open("run", "w") as file:
         file.write(nmr_shifts_input)


   os.chdir( path+"/nmr_shifts" )

   nnodes, ntasks, mem, time, parti, qos, prog, xtb_home, email, opt, xtb, xtb_hess = batch_settings( "nmr_shifts" )
   source_file = "batch_array"
   with open( "slurm_template", "w" ) as file:
      file.write( batch_file.format( NODES=nnodes, TASKS=ntasks, MEM=mem, TIME=time, PARTI=parti, QOS=qos, MAIL=email,                     \
                                     PROG=prog, XTB=xtb_home, MULTI=multi, CHARGE=charge, SOLV_XTB=solvent_XTB, XTB_SMODEL=XTB_solv_model, \
                                     OPT_BOOL=opt, XTB_BOOL=xtb, SOURCE=source_file, XTB_HESS=xtb_hess                                     ) )

   cpy_conformers = copy.deepcopy( conformers )
   comp_driver( path, "nmr_shifts", cpy_conformers )

   # Evaluate timings
   timings( path, "nmr_shifts", filter_results )


   os.chdir( path )
   with open( "filter_results.json", "w" ) as file:
      json.dump( filter_results, file, indent=4 )
   with open( "log", "a" ) as file:
      file.write( "nmr shifts done\n" )


   return
#-------------------------------------------------------------------------------




def nmr_couplings( path ):
#-------------------------------------------------------------------------------

   os.chdir(path)

   en_bm_list, nr_conf_G, nr_conf_BM = bm_eval( source, "sorting_couplings.out" )

   if nr_conf_G < nr_conf_BM:
      nr_conf = nr_conf_G
   else:
      nr_conf = nr_conf_BM
   print("\nChoosing lowest number of conformers for sorting process ...\n")

   # Load json file to write timings to it later on
   with open( "filter_results.json", "r" ) as file:
      filter_results=json.load( file )

   os.mkdir( path+"/nmr_couplings" )

   conformers = []
   for conf in range( 0, nr_conf ):
      if filter_results[en_bm_list[conf][0]]["filter_3"]["converged"] == "true" :
         os.mkdir( path+"/nmr_couplings/conformer-"+str(en_bm_list[conf][0]) )
         shutil.copy( path+"/filter_3/conformer-"+str(en_bm_list[conf][0])+"/run.xyz", path+"/nmr_couplings/conformer-"+str(en_bm_list[conf][0])+"/"+str(en_bm_list[conf][0])+".xyz" )
      conformers.append( int(en_bm_list[conf][0]) )


   os.chdir( path )
   conformers.sort()
   with open( "anmr_enso", "w" ) as file:
      file.write( "ONOFF NMR  CONF BW      Energy       Gsolv      mRRHO      gi\n" )
      for conf in conformers:
         for index in range( len(en_bm_list) ):
            if conf == int(en_bm_list[index][0]):
               file.write( "1  {0:>5}{1:>5}    {2:6.4f} {3:<13.7f} 0.0000000  0.0000000 1.000\n".format( str(en_bm_list[index][0]), str(en_bm_list[index][0]), float(en_bm_list[index][4])/100, float(en_bm_list[index][1]) ) )

   for conf in conformers:
      os.chdir( path+"/nmr_couplings/conformer-"+str(conf) )
      with open("run", "w") as file:
         file.write(nmr_couplings_input)


   os.chdir( path+"/nmr_couplings" )

   nnodes, ntasks, mem, time, parti, qos, prog, xtb_home, email, opt, xtb, xtb_hess = batch_settings( "nmr_couplings" )
   source_file = "batch_array"
   with open( "slurm_template", "w" ) as file:
      file.write( batch_file.format( NODES=nnodes, TASKS=ntasks, MEM=mem, TIME=time, PARTI=parti, QOS=qos, MAIL=email,                     \
                                     PROG=prog, XTB=xtb_home, MULTI=multi, CHARGE=charge, SOLV_XTB=solvent_XTB, XTB_SMODEL=XTB_solv_model, \
                                     OPT_BOOL=opt, XTB_BOOL=xtb, SOURCE=source_file, XTB_HESS=xtb_hess                                     ) )

   cpy_conformers = copy.deepcopy( conformers )
   comp_driver( path, "nmr_couplings", cpy_conformers )

   # Evaluate timings
   timings( path, "nmr_couplings", filter_results )


   os.chdir( path )
   with open( "filter_results.json", "w" ) as file:
      json.dump( filter_results, file, indent=4 )
   with open( "log", "a" ) as file:
      file.write( "nmr couplings done\n" )


   return
#-------------------------------------------------------------------------------




def nmr_evaluation( path ):
#-------------------------------------------------------------------------------

   os.chdir( path )

   with open( "filter_results.json", "r" ) as file:
      filter_results = json.load( file )

   os.mkdir( path+"/nmr_evaluation" )
   os.chdir( path+"/nmr_evaluation" )


   en_bm_list, nr_conf_G, nr_conf_BM = bm_eval( source, "sorting_couplings.out" )

   if nr_conf_G < nr_conf_BM:
      nr_conf = nr_conf_G
   else:
      nr_conf = nr_conf_BM
   print("\nChoosing lowest number of conformers for sorting process ...\n")

   conformers = []
   for conf in range( 0, nr_conf ):
      if filter_results[en_bm_list[conf][0]]["filter_3"]["converged"] == "true" :
         conformers.append( en_bm_list[conf][0] )


   if nmr_freq is None:
      print("\nError: experimental NMR frequency not set. Use option '--freq'\n")
      quit()
   if acidic_H is not None:
      aH = " ".join(acidic_H)+" XH acid atoms\n"
   else:
      aH = "XH acid atoms\n"
   solv = filter_results["Solvent"]
   anmrrc = aH+                                                            \
            "ENSO qm= ORCA mf= {0} lw= 1.0  J= on S= on T= 298.15\n"       \
            "TMS[acetone] kt3[SMD]/pcSseg-3//r2scan-3c[SMD]/def2-mTZVPP\n" \
            "  1  {1}    0.0     1\n"                                      \
            "  6  {2}    0.0     1\n".format( nmr_freq, References["KT3/pcSseg-3"][solv]["1H"], References["KT3/pcSseg-3"][solv]["13C"] )

   os.chdir( path+"/nmr_evaluation" )
   with open( ".anmrrc", "w" ) as file:
      file.write( anmrrc )


   # determine number of C and H (aka NMR active) atoms
   n_atoms, n_conformers=get_nr_atoms_conformers( path )
   extract_cart_coord( source, n_atoms, conformers_list=[ str(1) ] )
   active_atoms_list = []
   lnr=0
   anr=0
   with open( "1.xyz", "r" ) as file:
      for line in file:
         lnr += 1
         if lnr <= 2:
            continue
         anr+=1
         line.strip()
         line = line.split()
         if line[0] == "C" or line[0] == "H":
            active_atoms_list.append( anr )

   with open( "coord", "w" ) as file:
      file.write( "$coord\n" )
      with open( "1.xyz", "r" ) as file2:
         for line in file2:
            line.strip()
            if re.match( "^[A-Za-z]", line ):
               file.write( line+"\n" )
      file.write( "$end\n" )

   for conf in conformers:

      os.chdir( path+"/nmr_evaluation" )
      os.mkdir( path+"/nmr_evaluation/CONF"+str(conf) )
      os.mkdir( path+"/nmr_evaluation/CONF"+str(conf)+"/NMR" )
      shutil.copy( path+"/nmr_shifts/conformer-"+str(conf)+"/run.out", path+"/nmr_evaluation/CONF"+str(conf)+"/NMR/shifts.out" )
      shutil.copy( path+"/nmr_couplings/conformer-"+str(conf)+"/run.out", path+"/nmr_evaluation/CONF"+str(conf)+"/NMR/couplings.out" )

      os.chdir( path+"/nmr_evaluation/CONF"+str(conf)+"/NMR" )

      begin=False
      with open( "shifts.out", "r" ) as file:
         shifts_list = []
         for line in file:
            if "Nucleus  Element    Isotropic     Anisotropy" in line:
               begin=True
               continue
            if "---" in line:
               continue
            if begin:
               line = line.strip()
               line = line.split()
               if len(line) == 0:
                  break
               shifts_list.append( [ int(line[0]), line[2] ] )
      shifts_list.sort( key = lambda x: x[0] )
      with open( "nmrprop.dat", "w" ) as file:
         for i in range( 0, len(shifts_list) ):
            file.write( "{0:>4} {1}\n".format( shifts_list[i][0]+1, shifts_list[i][1] ) )
         file.write( "\n" )

      jcouplings_list = {}
      for atom in active_atoms_list:
         jcouplings_list[atom] = []
      block_size = len(active_atoms_list)
      block = 0
      with open( "couplings.out", "r" ) as file:
         begin=False
         for line in file:

            if "SUMMARY OF ISOTROPIC COUPLING CONSTANTS" in line:
               begin=True
               continue
            if "---" in line:
               continue

            if begin:
               # zero line is the column labeling
               if block == 0:
                  block += 1
                  continue
               if "Maximum memory used throughout" in line:
                  # end of Orca's J-coupling summary section is reached -> quit
                  break
               line = line.strip()
               line = line.split()
               c = 1
               for element in line:
                  # ignore the first two elements (Orca atom number, chem. element) of each block line
                  if c == 1 or c == 2:
                     c += 1
                  else:
                     jcouplings_list[int(line[0])+1].append( element )
               # block size reached -> reset value and start over with next block
               if block == block_size:
                  block = 0
                  continue
               block += 1


      with open( "nmrprop.dat", "a" ) as file:
         for atom in jcouplings_list.keys():
            m_element = 1
            for coupling in jcouplings_list[atom]:
               if m_element >= atom :
                  file.write( "{0:>4} {1:>4}   {2:<}\n".format( atom, m_element+1, coupling ) )
               m_element += 1



   os.chdir( path )

   return
#-------------------------------------------------------------------------------




#-------------------------------------------------------------------------------
#
#                                  M  A  I  N
#
#-------------------------------------------------------------------------------

if __name__ == "__main__":

   if fstep == "all":
      filter_1( source )
      packing( source, "filter_1" )
      filter_2( source )
      packing( source, "filter_2" )
      filter_3_driver( source )
      conf_gen( source )
      packing( source, "filter_3" )
      nmr_shifts( source )
      nmr_couplings( source )
      nmr_evaluation( source )
      packing( source, "nmr_shifts" )
      packing( source, "nmr_couplings" )


   if fstep == "filter":
      filter_1( source )
      packing( source, "filter_1" )
      filter_2( source )
      packing( source, "filter_2" )
      filter_3_driver( source )
      conf_gen( source )
      packing( source, "filter_3" )


   if fstep == "nmr":
      nmr_shifts( source )
      nmr_couplings( source )
      nmr_evaluation( source )
      packing( source, "nmr_shifts" )
      packing( source, "nmr_couplings" )


   if fstep == "1":
      filter_1( source )
      packing( source, "filter_1" )


   if fstep == "2":
      filter_2( source )
      packing( source, "filter_2" )


   if fstep == "3":
      filter_3_driver( source )
      packing( source, "filter_3" )


   if fstep == "sort":
      print("")
      if not os.path.isdir( source+"/filter_3" ):
         print("Unpacking data...")
         unpacking( source, "filter_3" )
      if os.path.isdir( source+"/sorting" ):
         shutil.rmtree( source+"/sorting" )
      print("Sorting ensemble conformers data...")
      conf_gen( source )
      print("Packing data...")
      packing( source, "filter_3" )
      print("")


   if fstep == "shifts":
      if not os.path.isdir( source+"/sorting" ):
         print("\n Error: data from sorting step not found\n")
         quit()
      nmr_shifts( source )
      packing( source, "nmr_shifts" )


   if fstep == "ssccs":
      if not os.path.isdir( source+"/sorting" ):
         print("\n Error: data from sorting step not found\n")
         quit()
      nmr_couplings( source )
      packing( source, "nmr_couplings" )


   if fstep == "eval":
      if not os.path.isdir( source+"/sorting" ):
         print("\n Error: data from sorting step not found\n")
         quit()
      print("")
      if not os.path.isdir( source+"/nmr_shifts" ):
         print("Unpacking NMR shifts data...")
         unpacking( source, "nmr_shifts" )
      if not os.path.isdir( source+"/nmr_couplings" ):
         print("Unpacking NMR couplings data...")
         unpacking( source, "nmr_couplings" )
      if os.path.isdir( source+"/nmr_evaluation" ):
         shutil.rmtree( source+"/nmr_evaluation" )
      print("Evaluating data...")            ; nmr_evaluation( source )
      print("Packing NMR shifts data...")    ; packing( source, "nmr_shifts" )
      print("Packing NMR couplings data...") ; packing( source, "nmr_couplings" )
      print("")


