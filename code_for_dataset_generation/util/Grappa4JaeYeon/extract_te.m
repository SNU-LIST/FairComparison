function te= extract_te(h,fname,eres)

        mdh1=readfirstmdh(fname);
        te1=mdh1.aushFreePara1;
        es=mdh1.aushFreePara2;
        te=te1:es:te1+es*(eres-1);
        te=.000001*te';
        
        if te>0
        else
            te =h.hdr.Meas.alTE;
            te = te(te>0);
            te = te(1:eres);
            te=.000001*te';
        end
end