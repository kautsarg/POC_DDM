# TITAN
This folder contains the data for pre-processing and processing of the data from Titan chips.
The array consists of 290 rows and 204 columns of sensors, where 1 temperature pixel is present avery 24 chemical pixels.

1. [ Lacewing Interface. ](#lacewing-interface)
2. [ Structure of the Experiment Folder. ](#experiment-folder)
3. [ Code Structure. ](#code-structure)
4. [ Preprocessing Workflow. ](#preprocessing-workflow)
5. [ Data. ](#data-folders)
6. [ How to use this repository: A (very quick) guide to GitHub. ](#github-guide)
   1. [ Connecting to GitHub via SSH. ](#github-guide-ssh)
   2. [ Creating a private branch of a public (or private) repo. ](#github-guide-private)

<a name="lacewing-interface"></a>
## Lacewing Interface 
1) _Init_ - reset memory on the chip. Check that it is electrically and chemically active. 
2) _Active Pixels_ - changes the Vref voltage from the minimum to the maximum. If there is a variation in the pixel output, 
this means that there exists a Vref value to bring the pixel in range. Generates file **\*find_active\*.bin**
3) _Calibrate Vref_ - Sweeps the Vref value in intervals of 100mV, counting the number of pixels in range to find the 
optimal value Vopt1. Then sweep Vref in steps of 10mV in the range Vopt1+-100mV to find Vref_optimal. 
4) _Evaluate pixels_ - generates file **\*evaluate_pixels\*.bin** . This stores information on the pixels that are 
active at any given Vref. If Vref is found manually, the experiment folder will have many such files. 
5) _Calibrate Vs_ - Find optimal Vs for each pixel. Generates file **\*vs_calibration\*.bin**
6) _Check Gain_ - Impose a known Vref variation = 58.594mV, to find the attenuation at each pixel. 
Generates file **\*gain\*.bin**
7) _Readout_ - Generates file **\*readout\*.bin**

<a name="experiment-folder"></a>
## Structure of the Experiment Folder
Data files in the experiment folder: 
- **\*find_active\*.bin**: reports the pixels for which a Vref can be found to bring them in range.
- **\*evaluate_pixels\*.bin**: reports pixels that are active at a given Vref. 
- **\*vs_calibration\*.bin**, reports the optimal Vs for each pixel.
- **\*gain\*.bin**. The data in this file is (N*M+Ag)\*Tg where N=number of rows, M=number of columns, 
Ag number of additional parameters per frame, Tg=number of frames for calibration. 
This records the pixels' response to a Vref variation = 58.594mV. Tg is 
  - Ag[0] = n_param: number of additional parameters (here, 9)
  - Ag[1] = vref: reference electrode voltage
  - Ag[2] = average output of pixels in well 1
  - Ag[3] = average output of pixels in well 2
  - Ag[>=4] = average output of pixels in well 3, ...
  - Ag[...] = optional additional parameters
- **\*readout\*.bin**. The data in this file is (N*M+A)\*T where N=number of rows, M=number of columns, 
A=number of additional parameters per frame, T=number of frames. Additional parameters are as follows:
  - A[0] = n_param: number of additional parameters (here, 9)
  - A[1] = time_stamp\*10: time vector
  - A[2] = average chem readout
  - A[3] = average temp readout
  - A[4] = temp_lin+100: average linearised temp readout
  - A[5] = vref: reference electrode voltage
  - A[6] = adc: ADC value 
  - A[7] = tam: ambient temperature
  - A[8] = tch: chamber temperature
  - A[>=9] = optional additional parameters 

A note on the version of the data generated - in the load_and_preprocessing function, 
the input variable data_version can be as follows. The list details the updates compared to the previous version. 
- "v01" version 0.1
  - A = 7 or 8
  - A[0] = time_sample
  - Ag = 6 or 10
  - Gain calibration has 4 frames (vrex x2, vref-60mV x2)
  - vref value is not stored in Ag. To know the value of vref in gain file, need to look at the log file of the readout. 
The value of Vref at the start of the readout is the same as frames 0-1 in the gain calibration. 
To this, subtract 60mV for frames 2-3, 120mV for frames 4-5 and 4V for frames 6-7

- "v02" version 0.2
  - A values updated to what is reported above: A[0]=num_params, A[1]=time_samples, ...
  
- "v03" version 0.3 since 16/12/2025
  - the gain file has 8 frames (vref x2, vref-60mV x2, vref-120mV x2, Vref-4V x2)

- "v04": version 0.4 (LATEST) since 11/02/2025
  - Ag values updated to the above: Ag[0] = num_params in the gain calibration file, Ag[1] = vref, ...

<a name="code-structure"></a>
## Code Structure
The files are structured as follows: 
- `functions.py` contains some generic base functions.
- `load_functions.py` has functions for loading .bin and .csv files.
- `preprocessing_functions.py` has function for preprocessing and filtering of the data.
- `Well.py` has the class definition for the well data
- `Experiment.py` has the class definition for the experiment data

<a name="preprocessing-workflow"></a>
## Preprocessing workflow
Find the link to the interactive workflow [HERE](https://miro.com/app/board/uXjVKx42EIY=/?share_link_id=330808356362)

<a name="data-folders"></a>
## Data 
Please **update this section** as new data becomes available. Include also all the information needed to use the data (number of wells, labels of each well, ...)

Request access to the data to the owner:
- Nick: [TB Data](https://imperiallondon-my.sharepoint.com/:f:/r/personal/nbm13_ic_ac_uk/Documents/Lacewing_Trials/23_London_TB/Samples?csf=1&web=1&e=v6Fvcy)
  6 wells x 2 chips per experiment.
  - Labels of the data: Look at TB_chipexpt.xlsx > sheet "chip_results" > column "RT-qLAMP (min)": if 0, negative
  - Samples are arranged as from the following table:

| **E1** |     | **E2** |      |
|--------|-----|--------|------|
| P      | P   | P      | P    |
| P      | N   | P      | P    |
| P      | N   | P?     | N    |

- Nick: [Covid quantification data](https://imperiallondon-my.sharepoint.com/personal/nbm13_ic_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fnbm13%5Fic%5Fac%5Fuk%2FDocuments%2FLacewing%5FTrials%2F24%5FCoV%5FQuantification&ct=1726069751585&or=OWA%2DNT%2DMail&cid=29a877c1%2De967%2D3269%2D8d98%2D0003f39d94f3&ga=1) this also includes a folder on linearisation, where Vref and Vtemp is modified to obtain the respective output signal
- Calista: [Colorectal Cancer BRAF p.V600E Data](https://imperiallondon-my.sharepoint.com/:f:/g/personal/cay18_ic_ac_uk/Eh0RxcjPPPhAvr1JBswuOCEBXWOrhPvIAyjIORJZdpyPOw?e=I6f7NG)
  6 well experiments, details on experiments to be updated soon
- Calista: [Water Control Data](https://imperiallondon-my.sharepoint.com/:f:/g/personal/cay18_ic_ac_uk/EsIBETuloUhAuayL3166msIBp2mVqygTWjUus2zXIvXUow?e=alZJO3)
  6 well experiments

<a name="github-guide"></a>
## How to use this repository: A (very quick) guide to GitHub
Depending on the IDE you are using, logging into your GitHub account may be different. Here is how to [connect to your GitHub account using SSH](#github-guide-ssh).
This method is particularly useful when using a [local repo of this project with multiple remote repos](#github-guide-private) (e.g. this one and your own personal remote). It will also work across different OSes and IDEs.
1) Copy the GitHub repository into your device: configure a local repository from a remote ``git clone ADDRESS-OF-REMOTE-REPO``\
   To clone this repository:
   - navigate to the location where you would like to copy the folder.
   - ``git clone https://github.com/cg3717/titan-processing-costanza.git``

2) Keep the branch up to date with the remote changes
   - **add** 
     ``git add FILE-NAME``\
     add all ``git add .``
   - **commit** ``git commit -m "MY-MESSAGE"``
   - **push** ``git push`` to synchronise any committed changes in the _current_ branch to the upstream branch (changes in the staging area are not synched)
   - **pull** ``git pull`` to bring changes from the upstream branch to the local branch \
     ``git fetch`` to check if there are changes \
     ``git status`` to check how local and remote branches compare in terms of commit history
   - typical workflow would be:
     ```
     git pull                               # update your local branch with changes from other users. 
                                            # Skip this ONLY if you are SURE that no one else is committing to your branch
     git add [file-to-add] (or) git add .   # add updates to current commit
     git git commit -m "my-message"         # commit changes. Add a MEANINGFUL DESCRIPTION to your commit 
     git push                               # push to the changes to the remote repo
     ```

3) Create a branch for your work and work on that branch: ``git checkout -b [your-bruch-name]``\
   Some useful commands for using **branches**:
   - ``git branch`` what branch am I using?
   - ``git branch BR_NAME`` create branch
   - ``git checkout BR_NAME`` use the new branch
   - ``git checkout -b BR_NAME`` create branch and switch to it
   - ``git merge --no-edit BR_NAME`` merge BR_NAME into main: when in main
   - ``git branch -d BR_NAME`` delete branch
   - typical workflow to work on a new-feature and add it to the main when it is ready: 
     ```
     git checkout -b new-feature  # create branch, switch to it
     git commit                   # work, work, work, ...; test; feature is ready
     git checkout main            # switch to main
     git merge new-feature        # merge work to main
     git branch -d new-feature    # remove branch
     ```

4) Keep your branch up to date with changes from the main:
     ```
     git checkout main        # go to main
     git pull                 # update te local version of the main 
     git checkout my-branch   # go bakc to my-branch
     git merge main           # merge work of the main into my-branch
     ```
   
    
<a name="github-guide-ssh"></a>
### Connecting to GitHub via SSH
Start from **STEP 4** if you have already generated an SSH key pair and linked it to your Github account.

1. **Generate an SSH key pair**
    ```bash
    ssh-keygen -t ed25519 -C "your_email@example.com"
    ```
   - If prompted, save the key in the default location (`~/.ssh/id_ed25519`). Note that `id_ed25519` is the file name (i.e. what the key will be called). 
   - Set a passphrase for extra security (optional, press Enter to leave blank).


2. **Add your SSH key to GitHub**

    Copy the public key to your clipboard:
    ```bash
    cat ~/.ssh/id_ed25519.pub
    ```
   - Go to your GitHub account: *Settings > SSH and GPG keys > New SSH key*.
   - Paste the public key into the text box and save.


3. **Test the SSH connection**
    ```bash
    ssh -T git@github.com
    ```
   - You should see a confirmation message like `Hi <username>! You've successfully authenticated`.

4. **Start the SSH agent (must be done every time the terminal is closed)**
    ```bash
    eval "$(ssh-agent -s)"
    ```

5. **Add your private key to the agent**
    ```bash
    ssh-add ~/.ssh/id_ed25519
    ```
   - You can now test your connection again (**STEP 3**) and continue using git as normal.


6. **Automate the SSH agent start (optional)**

    Add the following lines to your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`, could be different depending on your system/shell):
    ```bash
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_ed25519
    ```
    After saving the file, reload it:
    ```bash
    source ~/.bashrc  # Or source ~/.zshrc
    ```

With this setup, your SSH key will be securely stored, and git commands will automatically use the SSH connection.



---
<a name="github-guide-private"></a>
### Creating a private branch of a public (or private) repo

These instructions were adapted from [this resource](https://gist.github.com/mfbenitezp/5a49086a6c8333fc3b82e56b7892f7ee).
This gist describes how to create private branch (referred to as `downstream`) in a private repo (referred to as `origin`) of a shared (public or private) repository (referred to as `upstream`).

Make sure you are logged into your GitHub account via your IDE or [connect to it via SSH](#github-guide-ssh).

1. **Add remotes**

    Assuming you have already followed the instructions to clone the GitHub repo to your device and initialise the local repository, add the remote repos (`origin` and `upstream`).
    ```shell
    $ git remote add upstream git@github.com:<username>/public-repo.git
    $ git remote add origin git@github.com:<username>/private-repo.git
    ```
    You can find the names of the remotes (`public-repo.git` and `private-repo.git`) by looking at the GitHub URL.
    
    You can check your remotes by executing the following:
    ```
    $ git remote --verbose
    ```
    You should see something like this:
    
    ![gitremoteverbose](readme_files/gitremoteverbose.png)
    
    You can now use your private remote repo `origin` as you wish. Remember to specify which remote to use when push/pulling (`git push origin ...`, `git pull origin...`).
    
    The rest of these instructions deal with push/pulling to/from the shared repository `upstream`.
    
    If you get a `fatal: The current branch ... has no upstream branch.` you can use the `--set-upstream` tag to specify which remote the operation should target. There is also a way to do this automatically.

2. **Pulling changes**

    Switch to the branch you want to update (should be your main or master in your private repo, but can be any) and execute the following commands.
    
    ```shell
    git pull upstream main
    git merge --allow-unrelated-histories main
    git push origin main
    ```

3. **Pushing changes**
   1. **Create `develop` branch**
   
      You **must** create a new branch (referred to `develop`) in your private repo to push commits to the shared repo. If you don't, all personal files/data/tests/useless code will also be pushed. Merge conflicts will probably arise and cause a bunch of issues. 
   
       Create a new branch containing only the staged changes and switch to it (information on staging specific changes may be found elsewhere).

       ```shell
       $ git add ...
       $ git checkout -b develop
       $ git commit -m "Add information about your changes here"
       ```
    
       Push this commit to your private repo (not essential but good for tracking changes and record keeping).
    
       ```shell
       git push origin develop
       ```

   2. **Merge `develop` branch**

       Switch to the main branch of the shared repo (can also be a different branch, which is probably best practice in case the commit causes conflicts). Make sure that the branch you are pushing to is up-to-date with the shared remote.
    
       ```shell
       git switch main
       ```
    
       Merge the `develop` branch and push those changes.
    
       ```shell
       git merge develop
       git push upstream main
       ```
      
   3. **Delete `develop` branch (optional)**
      ```shell
      git branch -d develop
      git push origin --delete develop
      ```
